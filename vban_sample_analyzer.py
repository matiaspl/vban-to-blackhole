#!/usr/bin/env python3
"""
VBAN Sample Analyzer

Listens for VBAN UDP packets and analyzes the audio payload under several
decode hypotheses:
  - pcm16 signed/unsigned, little/big endian
  - pcm24 little/big endian
  - pcm32 little/big endian
  - float16 little/big endian
  - float32 little/big endian

For each candidate, computes per-channel statistics (RMS dBFS, peak dBFS,
mean, DC offset), detects NaN/Inf, and measures inter-channel correlations.
Ranks candidates by a heuristic score to help identify the correct format and
byte-order. Optionally override input channel count.

Usage examples:
  python vban_sample_analyzer.py --host 0.0.0.0 --port 6980 --once 20
  python vban_sample_analyzer.py --in-channels 8 --once 10
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import time
from typing import Iterable, Optional, Tuple

import numpy as np


VBAN_MAGIC = b"VBAN"
VBAN_HEADER_SIZE = 28


def bytes_to_hex(data: bytes, max_len: int = 64) -> str:
    view = data[:max_len]
    return " ".join(f"{b:02x}" for b in view) + (" ..." if len(data) > max_len else "")


def parse_naive_header_fields(header: bytes) -> dict:
    """Naive, best-effort parse of the 28-byte VBAN header.

    VBAN specs place 'VBAN' at 0..3, then bitfields. Here we expose raw bytes
    and basic interpretations that are commonly used in implementations:
      - sr_idx = header[4] & 0x1F  (lower 5 bits)
      - nbc    = header[5] & 0x1F  (channels-1)
      - format = (header[5] >> 5) & 0x07
      - bitres = header[6] & 0x1F  (implementation-dependent)
      - codec  = header[7]
      - stream name ~ header[8:24] (utf-8, nul-terminated)
    """
    if len(header) < VBAN_HEADER_SIZE:
        return {"valid": False}
    sr_byte = header[4]
    nbc_byte = header[5]
    bitres_byte = header[6]
    codec_byte = header[7]
    stream_name = header[8:24].split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
    frame_counter = int.from_bytes(header[24:28], "little", signed=False)
    return {
        "valid": True,
        "sr_raw": sr_byte,
        "nbc_raw": nbc_byte,
        "nbc_low5": nbc_byte & 0x1F,
        "channels_from_low5": (nbc_byte & 0x1F) + 1,
        "bitres_raw": bitres_byte,
        "codec_raw": codec_byte,
        "stream_name": stream_name,
        "frame_counter": frame_counter,
    }


def compute_channel_stats(audio: np.ndarray) -> dict:
    """Compute basic stats for audio shaped (frames, channels) float32 in [-1, 1]."""
    if audio.size == 0:
        return {
            "frames": 0,
            "channels": 0,
            "rms_dbfs": [],
            "peak_dbfs": [],
            "dc_offset": [],
            "nan_count": 0,
            "inf_count": 0,
            "corr_pairs": [],
        }
    frames, channels = audio.shape
    nan_count = int(np.isnan(audio).sum())
    inf_count = int(np.isinf(audio).sum())
    # Clamp to avoid invalid logs
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    # RMS per channel
    rms = np.sqrt(np.mean(np.square(audio), axis=0) + 1e-20)
    rms_dbfs = 20.0 * np.log10(rms)
    # Peak per channel
    peak = np.max(np.abs(audio), axis=0)
    peak_dbfs = 20.0 * np.log10(peak + 1e-20)
    # DC offset per channel
    dc = np.mean(audio, axis=0)
    # Correlation between adjacent channels (simple Pearson approx)
    corr_pairs = []
    if channels >= 2:
        for ch in range(0, min(channels - 1, 7)):
            a = audio[:, ch]
            b = audio[:, ch + 1]
            a = a - np.mean(a)
            b = b - np.mean(b)
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20)
            corr = float(np.dot(a, b) / denom)
            corr_pairs.append(((ch, ch + 1), corr))
    return {
        "frames": frames,
        "channels": channels,
        "rms_dbfs": rms_dbfs.tolist(),
        "peak_dbfs": peak_dbfs.tolist(),
        "dc_offset": dc.tolist(),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "corr_pairs": corr_pairs,
    }


def score_candidate(stats: dict) -> float:
    """Heuristic score: higher is better.

    Penalize NaN/Inf, all-silence, extreme clipping, huge DC offsets, and
    encourage RMS in a sensible range (-40dB..-3dB) with varied channels.
    """
    if stats["frames"] == 0 or stats["channels"] == 0:
        return -1e9
    rms = np.array(stats["rms_dbfs"], dtype=np.float32)
    peak = np.array(stats["peak_dbfs"], dtype=np.float32)
    dc = np.array(stats["dc_offset"], dtype=np.float32)
    score = 0.0
    score -= 5.0 * (stats["nan_count"] > 0)
    score -= 5.0 * (stats["inf_count"] > 0)
    # Reward rms in range
    for val in rms:
        if -40.0 <= val <= -3.0:
            score += 1.0
    # Penalize too low/high rms
        else:
            score -= 0.2
    # Penalize extreme peaks (clipping often)
    score -= float(np.sum(peak > -0.5)) * 0.1
    # Penalize DC offset
    score -= float(np.sum(np.abs(dc) > 0.02)) * 0.3
    return float(score)


def try_decode(payload: bytes, channels: int) -> list[tuple[str, str, str, dict, float]]:
    """Try multiple decode hypotheses and return [(fmt, endian, layout, stats, score)]."""
    candidates: list[tuple[str, str]] = [
        ("pcm16", "little"), ("pcm16", "big"),
        ("pcm16u", "little"), ("pcm16u", "big"),
        ("pcm24", "little"), ("pcm24", "big"),
        ("pcm32", "little"), ("pcm32", "big"),
        ("float16", "little"), ("float16", "big"),
        ("float32", "little"), ("float32", "big"),
    ]

    results: list[tuple[str, str, str, dict, float]] = []
    for fmt, endian in candidates:
        try:
            if fmt == "pcm16":
                dtype = "<i2" if endian == "little" else ">i2"
                a = np.frombuffer(payload, dtype=dtype)
                if a.size % channels:
                    continue
                frames = a.size // channels
                audio_i = (a.astype(np.float32) / 32768.0).reshape(frames, channels)
                audio_p = (a.astype(np.float32) / 32768.0).reshape(channels, frames).T
            elif fmt == "pcm16u":
                dtype = "<u2" if endian == "little" else ">u2"
                a = np.frombuffer(payload, dtype=dtype)
                if a.size % channels:
                    continue
                frames = a.size // channels
                ai = (a.astype(np.float32) - 32768.0) / 32768.0
                audio_i = ai.reshape(frames, channels)
                audio_p = ai.reshape(channels, frames).T
            elif fmt == "pcm24":
                bps = 3
                usable = (len(payload) // (channels * bps)) * (channels * bps)
                if usable == 0:
                    continue
                raw = np.frombuffer(payload[:usable], dtype=np.uint8).reshape(-1, 3)
                if endian == "little":
                    vals = (raw[:, 0].astype(np.int32)
                            | (raw[:, 1].astype(np.int32) << 8)
                            | (raw[:, 2].astype(np.int32) << 16))
                else:
                    vals = (raw[:, 2].astype(np.int32)
                            | (raw[:, 1].astype(np.int32) << 8)
                            | (raw[:, 0].astype(np.int32) << 16))
                sign_bit = 1 << 23
                vals = (vals ^ sign_bit) - sign_bit
                if vals.size % channels:
                    continue
                frames = vals.size // channels
                vf = vals.astype(np.float32) / 8388608.0
                audio_i = vf.reshape(frames, channels)
                audio_p = vf.reshape(channels, frames).T
            elif fmt == "pcm32":
                dtype = "<i4" if endian == "little" else ">i4"
                a = np.frombuffer(payload, dtype=dtype)
                if a.size % channels:
                    continue
                frames = a.size // channels
                vf = a.astype(np.float32) / 2147483648.0
                audio_i = vf.reshape(frames, channels)
                audio_p = vf.reshape(channels, frames).T
            elif fmt == "float16":
                dtype = "<f2" if endian == "little" else ">f2"
                a = np.frombuffer(payload, dtype=dtype)
                if a.size % channels:
                    continue
                frames = a.size // channels
                vf = a.astype(np.float32)
                audio_i = vf.reshape(frames, channels)
                audio_p = vf.reshape(channels, frames).T
            elif fmt == "float32":
                dtype = "<f4" if endian == "little" else ">f4"
                a = np.frombuffer(payload, dtype=dtype)
                if a.size % channels:
                    continue
                frames = a.size // channels
                vf = a.astype(np.float32, copy=False)
                audio_i = vf.reshape(frames, channels)
                audio_p = vf.reshape(channels, frames).T
            else:
                continue

            for layout, audio in (("interleaved", audio_i), ("planar", audio_p)):
                audio = np.clip(np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
                stats = compute_channel_stats(audio)
                score = score_candidate(stats)
                results.append((fmt, endian, layout, stats, score))
        except Exception:
            continue
    return results


async def analyze(args: argparse.Namespace) -> int:
    host = args.host
    port = args.port
    in_channels_override = args.in_channels
    max_packets = args.once

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.setblocking(False)

    print(f"Listening for VBAN on {host}:{port}")
    packets_seen = 0
    best_overall: Optional[tuple[str, str, float]] = None

    loop = asyncio.get_running_loop()
    try:
        while True:
            data, addr = await loop.sock_recvfrom(sock, 65536)
            if len(data) < VBAN_HEADER_SIZE or data[:4] != VBAN_MAGIC:
                if args.verbose:
                    print(f"Non-VBAN udp {len(data)} from {addr}")
                continue

            header = data[:VBAN_HEADER_SIZE]
            payload = data[VBAN_HEADER_SIZE:]

            hdr = parse_naive_header_fields(header)
            ch_header = hdr.get("channels_from_low5", None) if hdr.get("valid") else None
            in_ch = in_channels_override or ch_header or args.in_channels or 2

            if args.verbose or packets_seen % max(1, args.log_every) == 0:
                print("─" * 72)
                print(f"From {addr[0]}:{addr[1]} | bytes={len(data)} | payload={len(payload)}")
                print(f"Header hex: {bytes_to_hex(header, 64)}")
                if hdr.get("valid"):
                    print(f"  stream='{hdr['stream_name']}' sr_raw=0x{hdr['sr_raw']:02x} "
                          f"nbc_raw=0x{hdr['nbc_raw']:02x} (low5={hdr['nbc_low5']} -> ch={hdr['channels_from_low5']}) "
                          f"bitres=0x{hdr['bitres_raw']:02x} codec=0x{hdr['codec_raw']:02x} frame={hdr['frame_counter']}")
                else:
                    print("  invalid header length")
                print(f"Assuming input channels: {in_ch}")

            results = try_decode(payload, in_ch)
            if not results:
                if args.verbose:
                    print("No decoders produced frames for given channel count.")
                packets_seen += 1
                if max_packets and packets_seen >= max_packets:
                    break
                continue

            # Sort by score desc
            results.sort(key=lambda x: x[4], reverse=True)
            top = results[: min(6, len(results))]
            for fmt, endian, layout, stats, score in top:
                rms = stats["rms_dbfs"]
                peak = stats["peak_dbfs"]
                ch = stats["channels"]
                frames = stats["frames"]
                nan_i = stats["nan_count"]
                inf_i = stats["inf_count"]
                corr_str = ", ".join([f"{a}-{b}:{c:.2f}" for (a, b), c in stats["corr_pairs"][:4]])
                print(f"[{score:6.2f}] {fmt:8s}/{endian:6s}/{layout:10s} frames={frames:5d} ch={ch:2d} "
                      f"rms(dBFS)={np.mean(rms):6.1f} peak(dBFS)={np.max(peak):6.1f} "
                      f"nan={nan_i} inf={inf_i} corr_adj=({corr_str})")

            best_fmt, best_endian, best_layout, best_stats, best_score = top[0]
            if (best_overall is None) or (best_score > best_overall[2]):
                best_overall = (best_fmt, best_endian, best_score)

            packets_seen += 1
            if max_packets and packets_seen >= max_packets:
                break
    finally:
        sock.close()

    if best_overall:
        print("─" * 72)
        print(f"Best overall candidate: {best_overall[0]}/{best_overall[1]} score={best_overall[2]:.2f}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze VBAN audio payload to infer sample format and byte order.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    p.add_argument("--port", type=int, default=6980, help="UDP port to listen on")
    p.add_argument("--in-channels", type=int, default=None, help="Override input (sender) channel count; if not set, try header")
    p.add_argument("--once", type=int, default=20, help="Stop after this many VBAN packets (0=run forever)")
    p.add_argument("--log-every", type=int, default=5, help="Print header/details every N packets (when not verbose)")
    p.add_argument("--verbose", action="store_true", help="Verbose output for every packet")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        exit(asyncio.run(analyze(args)))
    except KeyboardInterrupt:
        exit(130)

