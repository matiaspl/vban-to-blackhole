#!/usr/bin/env python3
import argparse
import asyncio
import logging
import signal
import socket
import sys
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


VBAN_MAGIC = b"VBAN"
VBAN_HEADER_SIZE = 28  # Standard VBAN header length in bytes


class VUMeter:
    """VU meter with standard ballistics (~300 ms integration) and separate peak decay."""
    
    def __init__(self, channels: int, width: int = 40, sample_rate: int = 48000, time_constant_ms: float = 300.0,
                 peak_decay_db_per_s: float = 10.0):
        self.channels = channels
        self.width = width
        self.sample_rate = max(1, sample_rate)
        self.tau_s = max(1e-3, time_constant_ms / 1000.0)
        # Ballistic levels (linear, 0..1)
        self.ballistic_lin = [0.0] * channels
        self.levels = [0.0] * channels
        self.peak_hold = [0.0] * channels
        self.last_decay_time = time.monotonic()
        self.peak_decay_db_per_s = max(0.1, peak_decay_db_per_s)
        
    def update(self, audio_data: np.ndarray):
        if audio_data.size == 0:
            return
        frames = audio_data.shape[0]
        # One-pole IIR on per-block RMS to approximate 300 ms VU ballistics
        # alpha = 1 - exp(-dt / tau)
        dt = frames / float(self.sample_rate)
        alpha = 1.0 - np.exp(-dt / self.tau_s)
        for ch in range(min(self.channels, audio_data.shape[1])):
            channel_data = audio_data[:, ch]
            rms_block = float(np.sqrt(np.mean(np.square(channel_data)) + 1e-20))
            prev = self.ballistic_lin[ch]
            new_val = prev + alpha * (rms_block - prev)
            self.ballistic_lin[ch] = new_val
            self.levels[ch] = new_val
            # Instantaneous peak capture (absolute)
            peak_inst = float(np.max(np.abs(channel_data)))
            if peak_inst > self.peak_hold[ch]:
                self.peak_hold[ch] = peak_inst
                
    def draw(self) -> str:
        """Draw ASCII VU meter"""
        lines = []
        lines.append("─" * (self.width + 20))
        
        for ch in range(self.channels):
            level = self.levels[ch]
            peak = self.peak_hold[ch]
            
            # Convert to dB (relative to full scale)
            if level > 0:
                db = 20 * np.log10(level)
            else:
                db = -60
                
            if peak > 0:
                peak_db = 20 * np.log10(peak)
            else:
                peak_db = -60
                
            # Normalize to 0-1 range for bar
            bar_level = max(0.0, min(1.0, (db + 60.0) / 60.0))
            bar_width = int(bar_level * self.width)

            # Compute peak position on the same scale
            peak_level = max(0.0, min(1.0, (peak_db + 60.0) / 60.0))
            peak_pos = int(peak_level * (self.width - 1))

            # Build bar with intensity shading
            bar_chars = [" "] * self.width
            for i in range(self.width):
                if i < bar_width:
                    if i < int(bar_width * 0.7):
                        bar_chars[i] = "█"
                    elif i < int(bar_width * 0.9):
                        bar_chars[i] = "▓"
                    else:
                        bar_chars[i] = "░"

            # Overlay peak marker on the bar
            if 0 <= peak_pos < self.width:
                if bar_chars[peak_pos] == " ":
                    bar_chars[peak_pos] = "│"
                else:
                    bar_chars[peak_pos] = "┆"

            bar = "".join(bar_chars)
                    
            # Format channel info
            ch_info = f"Ch{ch+1:2d}"
            level_info = f"{db:6.1f}dB"
            peak_info = f"P:{peak_db:6.1f}dB"
            
            line = f"{ch_info} [{bar}] {level_info} {peak_info}"
            lines.append(line)
            
        lines.append("─" * (self.width + 20))
        return "\n".join(lines)
        
    def decay_peaks(self):
        """Apply decay to peak hold using configurable dB/s rate."""
        now = time.monotonic()
        dt = now - self.last_decay_time
        if dt <= 0:
            return
        # Multiply by factor corresponding to decay of peak_decay_db_per_s over dt seconds
        factor = 10.0 ** (-(self.peak_decay_db_per_s / 20.0) * dt)
        for i in range(self.channels):
            self.peak_hold[i] *= factor
        self.last_decay_time = now


class WavWriter:
    def __init__(self, path: str, samplerate: int, channels: int, dtype: str):
        """dtype: 'float32', 'int16', or 'int32'"""
        subtype = {
            'float32': 'FLOAT',
            'int16': 'PCM_16',
            'int32': 'PCM_32',
        }.get(dtype, 'FLOAT')
        format = 'WAV'
        self.path = path
        self.file = sf.SoundFile(path, mode='w', samplerate=samplerate, channels=channels, subtype=subtype, format=format)
        self.dtype = dtype
        self.closed = False

    def write(self, frames: np.ndarray):
        if not self.closed:
            self.file.write(frames)

    def close(self):
        if not self.closed:
            try:
                self.file.flush()
                self.file.close()
            finally:
                self.closed = True


async def audio_playback(
    queue: asyncio.Queue,
    sample_rate: int,
    channels: int,
    output_device: int | str,
    logger: logging.Logger,
    stop_event: asyncio.Event,
    blocksize_frames: int = 0,
    output_dtype: str = "float32",
    wav_writer: "WavWriter | None" = None,
):
    """Continuously pull numpy float32 frames from queue and write to output device.

    Expects queue items shaped (N, channels), dtype=float32 in range [-1.0, 1.0].
    """
    stream = sd.OutputStream(
        device=output_device,
        samplerate=sample_rate,
        channels=channels,
        dtype=output_dtype,
        blocksize=blocksize_frames,
        latency="low",
    )
    
    try:
        stream.start()
        logger.info("Audio output: device='%s', sr=%d Hz, ch=%d", output_device, sample_rate, channels)
        
        while not stop_event.is_set():
            try:
                block = await asyncio.wait_for(queue.get(), timeout=0.02)
                if block is None:
                    break
                if output_dtype == "float32":
                    out = block.astype(np.float32, copy=False)
                    stream.write(out)
                    if wav_writer:
                        wav_writer.write(out)
                elif output_dtype == "int16":
                    out = np.clip(block * 32767.0, -32768.0, 32767.0).astype(np.int16)
                    stream.write(out)
                    if wav_writer:
                        wav_writer.write(out)
                elif output_dtype == "int32":
                    out = np.clip(block * 2147483647.0, -2147483648.0, 2147483647.0).astype(np.int32)
                    stream.write(out)
                    if wav_writer:
                        wav_writer.write(out)
                else:
                    out = block.astype(np.float32, copy=False)
                    stream.write(out)
                    if wav_writer:
                        wav_writer.write(out)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error("Audio playback error: %s", exc)
                break
    finally:
        try:
            stream.stop()
        finally:
            stream.close()


def find_output_device(name_or_index: str | int) -> tuple[int, str]:
    devices = sd.query_devices()
    if isinstance(name_or_index, int):
        try:
            dev = devices[name_or_index]
            return name_or_index, dev["name"]
        except Exception:
            raise ValueError(f"No output device at index {name_or_index}")

    # Try exact name match first
    for idx, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) > 0 and dev.get("name") == name_or_index:
            return idx, dev["name"]

    # Fallback to substring (case-insensitive)
    low = name_or_index.lower()
    for idx, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) > 0 and low in dev.get("name", "").lower():
            return idx, dev["name"]

    raise ValueError(f"Output device not found matching '{name_or_index}'. Use --list-devices to inspect.")


def list_output_devices() -> str:
    lines = ["Available output devices (PortAudio):"]
    for idx, dev in enumerate(sd.query_devices()):
        max_out = dev.get("max_output_channels", 0)
        if max_out > 0:
            lines.append(f"  [{idx:2d}] {dev.get('name')}  (max_out_ch={max_out})")
    return "\n".join(lines)


async def run(args) -> int:
    logger = logging.getLogger("vban-to-blackhole")
    logger.setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)

    if args.list_devices:
        print(list_output_devices())
        return 0

    try:
        device_index, device_name = find_output_device(args.output_device)
    except Exception as exc:
        logger.error("%s", exc)
        logger.info("Tip: run with --list-devices to see available outputs")
        return 2

    # Shared queue for decoded audio blocks
    queue: asyncio.Queue = asyncio.Queue(maxsize=args.buffer)

    # Stop event and signal handling
    stop_event = asyncio.Event()
    
    def _handle_signal(*_):
        logger.info("Received termination signal, shutting down...")
        stop_event.set()
        # Don't stop the loop - let it complete naturally
    
    # Use a more robust signal handling approach
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handle_signal)
                logger.info("Signal handler installed for %s", sig)
            except NotImplementedError:
                logger.warning("Signal handler not available for %s", sig)
                pass
    except Exception as e:
        logger.warning("Could not install signal handlers: %s", e)
    
    # Always install fallback signal handlers as backup
    import signal as signal_module
    def fallback_handler(sig, frame):
        logger.info("Received signal %s via fallback handler", sig)
        stop_event.set()
    signal_module.signal(signal_module.SIGINT, fallback_handler)
    signal_module.signal(signal_module.SIGTERM, fallback_handler)
    logger.info("Fallback signal handlers installed")
    
    # Add a signal monitoring task that runs in the background
    async def monitor_signals():
        """Monitor for stop event and exit if needed"""
        while not stop_event.is_set():
            await asyncio.sleep(0.1)  # Check every 100ms
        
        logger.info("Stop event detected, initiating shutdown...")
        # Don't force exit - let the cleanup happen naturally
    
    signal_monitor_task = asyncio.create_task(monitor_signals())

    async def udp_receive_decode():
        """Receive VBAN UDP packets and push decoded PCM float32 blocks to queue."""
        logger.info("Listening for VBAN on %s:%d", args.listen_ip, args.listen_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.listen_ip, args.listen_port))
        sock.setblocking(False)

        vu_meter = VUMeter(args.channels, sample_rate=args.sample_rate)
        last_vu_update = 0.0
        last_header_log = 0.0

        # Optional dump-to-file for N seconds
        # If --dump-raw is set, writes raw VBAN audio payload bytes (post-header) with no conversion
        # Otherwise writes decoded interleaved samples in selected --dump-format
        dump_enabled = args.dump_seconds is not None and args.dump_seconds > 0 and args.dump_file
        dump_target_frames = int((args.dump_seconds or 0) * args.sample_rate) if dump_enabled else 0
        dump_written_frames = 0
        dump_fh = None
        if dump_enabled:
            try:
                dump_fh = open(args.dump_file, "wb")
                if args.dump_raw:
                    logger.info("Dumping RAW VBAN payload to %s for ~%s seconds (%d frames target)", args.dump_file, args.dump_seconds, dump_target_frames)
                else:
                    logger.info("Dumping decoded audio to %s for ~%s seconds (%d frames target)", args.dump_file, args.dump_seconds, dump_target_frames)
            except Exception as exc:
                logger.error("Could not open dump file '%s': %s", args.dump_file, exc)
                dump_enabled = False
        
        loop = asyncio.get_running_loop()
        # Assemble consistent device block sizes to reduce jitter/warble
        device_blocksize = max(1, args.device_blocksize)
        jitter_frames = int(max(0, args.jitter_ms) * args.sample_rate / 1000)
        assemble = np.empty((0, args.channels), dtype=np.float32)
        started_playback = False
        try:
            while not stop_event.is_set():
                try:
                    data, addr = await loop.sock_recvfrom(sock, 65536)
                except (BlockingIOError, asyncio.TimeoutError):
                    await asyncio.sleep(0.001)
                    continue
                except Exception as exc:
                    logger.error("UDP receive error: %s", exc)
                    await asyncio.sleep(0.05)
                    continue

                if len(data) < VBAN_HEADER_SIZE or data[:4] != VBAN_MAGIC:
                    continue
                # Get source IP and port from socket
                src_ip, src_port = addr
                # Parse VBAN audio header fields (spec-compliant indexes)
                # SR, nbs (samples-1), nbc (channels-1), bit (datatype+codec)
                format_sr = data[4]
                format_nbs = data[5]
                format_nbc = data[6]
                format_bit = data[7]
                stream_name = data[8:24].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
                frame_counter = int.from_bytes(data[24:28], "little", signed=False)

                # Samples per frame and channels from header: stored as (value-1)
                nbc_raw = int(format_nbc)
                nbs_raw = int(format_nbs)
                hdr_channels = nbc_raw + 1
                hdr_samples_per_frame = nbs_raw + 1
                if hdr_channels < 1:
                    hdr_channels = 1
                elif hdr_channels > 256:
                    hdr_channels = 256
                if hdr_samples_per_frame < 1:
                    hdr_samples_per_frame = 1
                elif hdr_samples_per_frame > 256:
                    hdr_samples_per_frame = 256

                # Occasionally log parsed header details to help debugging
                now_ts = time.monotonic()
                if now_ts - last_header_log > 1.0:
                    datatype_index = int(format_bit & 0x07)
                    reserved_bit = int((format_bit >> 3) & 0x01)
                    codec_index = int((format_bit >> 4) & 0x0F)
                    logger.info(
                        "VBAN hdr: stream='%s' sr_idx=0x%02x nbs=%d nbc=%d bit=0x%02x (dt=%d codec=%d%s) frame=%d",
                        stream_name,
                        format_sr,
                        hdr_samples_per_frame,
                        hdr_channels,
                        format_bit,
                        datatype_index,
                        codec_index,
                        ", reserved=1" if reserved_bit else "",
                        frame_counter,
                    )
                    last_header_log = now_ts

                # Determine decoder from header bit-field by default; allow explicit CLI override
                codec_value = int(format_bit) & 0xF0
                datatype_index = int(format_bit) & 0x07
                if codec_value != 0x00:
                    if args.verbose:
                        logger.warning("Non-PCM VBAN codec (0x%02x) not supported; dropping packet", codec_value)
                    continue

                header_decoder = None
                header_bps = None
                if datatype_index == 0x00:
                    header_decoder, header_bps = "pcm8", 1
                elif datatype_index == 0x01:
                    header_decoder, header_bps = "pcm16", 2
                elif datatype_index == 0x02:
                    header_decoder, header_bps = "pcm24", 3
                elif datatype_index == 0x03:
                    header_decoder, header_bps = "pcm32", 4
                elif datatype_index == 0x04:
                    header_decoder, header_bps = "float32", 4
                elif datatype_index == 0x05:
                    if args.verbose:
                        logger.warning("FLOAT64 payload not supported; dropping packet")
                    continue
                else:
                    if args.verbose:
                        logger.warning("%d-bit packed payload (index=0x%02x) not supported; dropping packet", 12 if datatype_index==0x06 else 10, datatype_index)
                    continue

                # CLI override (if provided) takes precedence
                decoder = args.input_format or header_decoder
                if decoder == "pcm8":
                    bps = 1
                elif decoder == "pcm16":
                    bps = 2
                elif decoder == "pcm24":
                    bps = 3
                elif decoder == "pcm32":
                    bps = 4
                elif decoder == "float32":
                    bps = 4
                else:
                    decoder = header_decoder
                    bps = header_bps

                # Determine effective header size (some senders use 32 bytes). Try 28 then 32, choose one that fits frames.
                header_sizes = [VBAN_HEADER_SIZE, 32] if len(data) >= 32 else [VBAN_HEADER_SIZE]
                chosen_header = None
                chosen_payload = None

                # We'll select in_channels below; start with header suggestion if sensible
                in_channels = args.in_channels if args.in_channels is not None else None

                for hs in header_sizes:
                    payload = data[hs:]
                    if not payload:
                        continue

                    # Prefer spec header values when --in-channels is not provided
                    if in_channels is None:
                        in_channels_candidate = hdr_channels
                        bpf_hdr = in_channels_candidate * bps
                        expected_bytes_hdr = hdr_samples_per_frame * bpf_hdr if bpf_hdr else 0
                        if (
                            bpf_hdr > 0
                            and hdr_samples_per_frame > 0
                            and hdr_samples_per_frame <= 256
                            and len(payload) >= expected_bytes_hdr
                            and (len(payload) % bpf_hdr) == 0
                        ):
                            frames = len(payload) // bpf_hdr
                            if frames > 0 and frames <= 256:
                                chosen_header = hs
                                chosen_payload = payload[: expected_bytes_hdr if expected_bytes_hdr > 0 else len(payload)]
                                in_channels = in_channels_candidate
                                break
                        # If header doesn't match payload cleanly, try inference as fallback
                        best_ch = None
                        best_remainder = None
                        best_frames = 0
                        for ch in range(1, 33):
                            bpf = ch * bps
                            if bpf == 0:
                                continue
                            frames = len(payload) // bpf
                            remainder = len(payload) % bpf
                            if frames == 0 or frames > 256:
                                continue
                            if (
                                best_remainder is None
                                or remainder < best_remainder
                                or (remainder == best_remainder and frames > best_frames)
                            ):
                                best_remainder = remainder
                                best_ch = ch
                                best_frames = frames
                                if remainder == 0 and ch in (2, 4, 6, 8, 16):
                                    break
                        in_channels_candidate = best_ch if best_ch is not None else hdr_channels
                    else:
                        in_channels_candidate = in_channels

                    bytes_per_frame_src = in_channels_candidate * bps
                    usable = (len(payload) // bytes_per_frame_src) * bytes_per_frame_src
                    frames = usable // bytes_per_frame_src if bytes_per_frame_src else 0
                    if frames > 0 and frames <= 256:
                        chosen_header = hs
                        chosen_payload = payload[:usable]
                        in_channels = in_channels_candidate
                        break

                if chosen_payload is None:
                    # Could not find a clean fit; fall back to original header size with best-effort truncation using header channels
                    payload = data[VBAN_HEADER_SIZE:]
                    if not payload:
                        continue
                    if in_channels is None:
                        in_channels = hdr_channels
                    bytes_per_frame_src = in_channels * bps
                    usable = (len(payload) // bytes_per_frame_src) * bytes_per_frame_src
                    if usable == 0:
                        continue
                    chosen_payload = payload[:usable]
                    chosen_header = VBAN_HEADER_SIZE

                # No auto-format detection; rely on explicit --input-format and --byte-order

                # Decode into float32 [-1, 1], shape (frames, in_channels)
                if decoder == "pcm16":
                    # Strictly interpret as linear PCM signed 16-bit, interleaved
                    dtype16 = "<i2" if (args.byte_order or "little") == "little" else ">i2"
                    audio_i16 = np.frombuffer(chosen_payload, dtype=dtype16)
                    # Verify frame divisibility
                    if audio_i16.size % in_channels != 0:
                        # Truncate to whole frames
                        whole = (audio_i16.size // in_channels) * in_channels
                        audio_i16 = audio_i16[:whole]
                    frames = audio_i16.size // in_channels
                    # Convert to float32 in [-1,1]; use 32768.0 to map full scale negative exactly
                    audio = (audio_i16.astype(np.float32) / 32768.0).reshape(frames, in_channels)
                elif decoder == "pcm16u":
                    dtypeu16 = "<u2" if (args.byte_order or "little") == "little" else ">u2"
                    audio_u16 = np.frombuffer(chosen_payload, dtype=dtypeu16)
                    if audio_u16.size % in_channels != 0:
                        whole = (audio_u16.size // in_channels) * in_channels
                        audio_u16 = audio_u16[:whole]
                    frames = audio_u16.size // in_channels
                    # Map unsigned 16-bit (0..65535) to float32 (-1..1)
                    audio = (((audio_u16.astype(np.float32) - 32768.0) / 32768.0)
                             .reshape(frames, in_channels))
                elif decoder == "pcm24":
                    raw = np.frombuffer(chosen_payload, dtype=np.uint8).reshape(-1, 3)
                    if (args.byte_order or "little") == "little":
                        vals = (raw[:, 0].astype(np.int32)
                                | (raw[:, 1].astype(np.int32) << 8)
                                | (raw[:, 2].astype(np.int32) << 16))
                    else:
                        vals = (raw[:, 2].astype(np.int32)
                                | (raw[:, 1].astype(np.int32) << 8)
                                | (raw[:, 0].astype(np.int32) << 16))
                    sign_bit = 1 << 23
                    vals = (vals ^ sign_bit) - sign_bit
                    # Ensure multiple of channels
                    if vals.size % in_channels != 0:
                        whole = (vals.size // in_channels) * in_channels
                        vals = vals[:whole]
                    frames = vals.size // in_channels
                    audio = (vals.astype(np.float32) / 8388608.0).reshape(frames, in_channels)
                elif decoder == "pcm32":
                    dtype32 = "<i4" if (args.byte_order or "little") == "little" else ">i4"
                    audio_i32 = np.frombuffer(chosen_payload, dtype=dtype32)
                    if audio_i32.size % in_channels != 0:
                        whole = (audio_i32.size // in_channels) * in_channels
                        audio_i32 = audio_i32[:whole]
                    frames = audio_i32.size // in_channels
                    audio = (audio_i32.astype(np.float32) / 2147483648.0).reshape(frames, in_channels)
                elif decoder == "float16":
                    dtypef2 = "<f2" if (args.byte_order or "little") == "little" else ">f2"
                    a = np.frombuffer(chosen_payload, dtype=dtypef2)
                    if a.size % in_channels != 0:
                        whole = (a.size // in_channels) * in_channels
                        a = a[:whole]
                    audio = a.reshape(-1, in_channels).astype(np.float32)
                    frames = audio.shape[0]
                elif decoder == "float32":
                    dtypef = "<f4" if (args.byte_order or "little") == "little" else ">f4"
                    a = np.frombuffer(chosen_payload, dtype=dtypef)
                    if a.size % in_channels != 0:
                        whole = (a.size // in_channels) * in_channels
                        a = a[:whole]
                    audio = a.reshape(-1, in_channels).astype(np.float32, copy=False)
                    frames = audio.shape[0]
                elif decoder == "pcm8":
                    audio_u8 = np.frombuffer(chosen_payload, dtype=np.uint8)
                    if audio_u8.size % in_channels != 0:
                        whole = (audio_u8.size // in_channels) * in_channels
                        audio_u8 = audio_u8[:whole]
                    frames = audio_u8.size // in_channels
                    audio = (((audio_u8.astype(np.float32) - 128.0) / 128.0).reshape(frames, in_channels))
                else:
                        continue
                    
                # Protect against out-of-range values due to sender mismatch
                np.clip(audio, -1.0, 1.0, out=audio)

                # Ensure writable contiguous float32 buffer for safe processing
                audio = np.array(audio, dtype=np.float32, copy=True, order="C")

                # Map channels from source to desired output size
                # If the sender has fewer channels than requested, place them in the first channels and zero the rest.
                # If the sender has more channels, truncate to the requested count.
                if in_channels != args.channels:
                    if in_channels < args.channels:
                        out = np.zeros((frames, args.channels), dtype=np.float32)
                        out[:, :min(in_channels, args.channels)] = audio[:, :min(in_channels, args.channels)]
                        audio = out
                    else:
                        audio = audio[:, :args.channels]

                # Apply gain and sanitize values
                if args.gain != 1.0:
                    audio = audio * args.gain
                audio = np.clip(audio, -1.0, 1.0)
                audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

                # Optional: write to dump file
                if dump_enabled and dump_fh is not None and dump_written_frames < dump_target_frames:
                    try:
                        if args.dump_raw:
                            # Write raw VBAN audio payload bytes (full frames) with no conversion
                            # Use chosen_payload so frame counting/seconds is correct
                            frames_to_write = min(frames, dump_target_frames - dump_written_frames)
                            if frames_to_write > 0:
                                bytes_per_frame_out = in_channels * bps
                                raw_bytes = chosen_payload[: frames_to_write * bytes_per_frame_out]
                                dump_fh.write(raw_bytes)
                                dump_written_frames += frames_to_write
                        else:
                            # Write decoded interleaved samples in selected format
                            frames_to_write = min(frames, dump_target_frames - dump_written_frames)
                            if frames_to_write > 0:
                                block = audio[:frames_to_write, :]
                                fmt = args.dump_format
                                if fmt == "f32le":
                                    dump_fh.write(block.astype("<f4", copy=False).tobytes())
                                elif fmt == "f16le":
                                    dump_fh.write(block.astype("<f2").tobytes())
                                elif fmt == "i16le":
                                    i16 = np.clip(np.rint(block * 32767.0), -32768, 32767).astype("<i2")
                                    dump_fh.write(i16.tobytes())
                                elif fmt == "i24le":
                                    i24 = np.clip(np.rint(block * 8388607.0), -8388608, 8388607).astype(np.int32)
                                    b0 = (i24 & 0xFF).astype(np.uint8)
                                    b1 = ((i24 >> 8) & 0xFF).astype(np.uint8)
                                    b2 = ((i24 >> 16) & 0xFF).astype(np.uint8)
                                    packed = np.stack([b0, b1, b2], axis=-1).reshape(-1)
                                    dump_fh.write(packed.tobytes())
                                elif fmt == "i32le":
                                    i32 = np.clip(np.rint(block * 2147483647.0), -2147483648, 2147483647).astype("<i4")
                                    dump_fh.write(i32.tobytes())
                                else:
                                    dump_fh.write(block.astype("<f4", copy=False).tobytes())
                                dump_written_frames += frames_to_write
                        if dump_written_frames >= dump_target_frames:
                            dump_fh.flush()
                            dump_fh.close()
                            dump_fh = None
                            logger.info("Dump complete: wrote %d frames to %s", dump_written_frames, args.dump_file)
                            dump_enabled = False
                    except Exception as exc:
                        logger.error("Dump write error: %s", exc)
                        try:
                            dump_fh.close()
                        except Exception:
                            pass
                        dump_fh = None
                        dump_enabled = False
                            
                # Update VU and screen (if not verbose)
                if not args.verbose:
                    vu_meter.update(audio)
                    now = time.monotonic()
                    if now - last_vu_update > 0.1:
                        print("\033[2J\033[H")
                        print(f"VBAN: {stream_name} ({src_ip}:{src_port}) - {args.channels} channels @ {args.sample_rate} Hz")
                        print(vu_meter.draw())
                        vu_meter.decay_peaks()
                        last_vu_update = now

                # Assemble fixed-size blocks before enqueue
                if audio.shape[0] < device_blocksize:
                    assemble = np.concatenate([assemble, audio], axis=0)
                else:
                    assemble = np.concatenate([assemble, audio], axis=0)
                # Start after jitter buffer is accumulated
                if not started_playback:
                    if assemble.shape[0] >= max(device_blocksize, jitter_frames):
                        started_playback = True
                # Push out in device-blocksize chunks once started
                if started_playback:
                    while assemble.shape[0] >= device_blocksize:
                        out_block = assemble[:device_blocksize, :]
                        assemble = assemble[device_blocksize:, :]
                        try:
                            queue.put_nowait(out_block)
                        except asyncio.QueueFull:
                            try:
                                _ = queue.get_nowait()
                                queue.put_nowait(out_block)
                            except Exception:
                                pass
                    # If playback queue is starving, optionally pad one block of zeros to keep clock steady
                    if args.starve_fill and queue.empty() and assemble.shape[0] == 0:
                        try:
                            queue.put_nowait(np.zeros((device_blocksize, args.channels), dtype=np.float32))
                        except Exception:
                            pass
        finally:
            # Flush any remaining assembled frames
            try:
                if assemble is not None and assemble.shape[0] > 0:
                    try:
                        queue.put_nowait(assemble)
                    except asyncio.QueueFull:
                        try:
                            _ = queue.get_nowait()
                            queue.put_nowait(assemble)
                        except Exception:
                            pass
            except Exception:
                pass
            sock.close()

    # Optional WAV mirror writer
    wav_writer = None
    output_dtype = 'float32'
    if args.wav_path:
        if args.wav_dtype:
            output_dtype = args.wav_dtype
        try:
            wav_writer = WavWriter(args.wav_path, samplerate=args.sample_rate, channels=args.channels, dtype=output_dtype)
            logger.info("WAV mirror enabled: %s (%s, %d ch)", args.wav_path, output_dtype, args.channels)
        except Exception as exc:
            logger.error("Could not open WAV file '%s': %s", args.wav_path, exc)
            wav_writer = None

    # Start tasks
    receiver_task = asyncio.create_task(udp_receive_decode())
    playback_task = asyncio.create_task(
        audio_playback(
            queue=queue,
            sample_rate=args.sample_rate,
            channels=args.channels,
            output_device=device_index,
            logger=logger,
            stop_event=stop_event,
            blocksize_frames=max(1, args.device_blocksize),
            output_dtype=output_dtype,
            wav_writer=wav_writer,
        )
    )

    try:
        await stop_event.wait()
    except Exception as exc:
        logger.error("Playback stopped due to error: %s", exc)
        return 1
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        stop_event.set()
        for task in (receiver_task, playback_task, signal_monitor_task):
            if not task.done():
                task.cancel()
        for task in (receiver_task, playback_task, signal_monitor_task):
            try:
                await asyncio.wait_for(task, timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        # Close WAV file if open
        try:
            if wav_writer:
                wav_writer.close()
        except Exception:
            pass
        await asyncio.sleep(0.05)
        logger.info("Shutdown complete")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Receive VBAN audio over UDP and play to CoreAudio output (e.g., BlackHole 16ch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--listen-ip", default="0.0.0.0", help="IP address to bind for UDP VBAN packets")
    p.add_argument("--listen-port", type=int, default=6980, help="UDP port to listen on")
    p.add_argument("--sample-rate", type=int, default=48000, help="Audio sample rate (Hz)")
    p.add_argument("--channels", type=int, default=16, help="Number of audio channels")
    p.add_argument("--output-device", default="BlackHole 16ch", help="Output device name or index")
    p.add_argument("--buffer", type=int, default=16, help="Internal frame queue size (audio blocks)")
    p.add_argument("--in-channels", type=int, default=None, help="Override input (sender) channel count; fallback to VBAN header if not set")
    p.add_argument("--input-format", choices=["pcm8","pcm16","pcm16u","pcm24","pcm32","float16","float32"], default=None, help="Sender sample format; if unset, defaults to pcm16")
    p.add_argument("--byte-order", choices=["little","big"], default="little", help="Byte order of sender samples")
    p.add_argument("--dump-seconds", type=float, default=None, help="Dump decoded interleaved samples for N seconds to --dump-file")
    p.add_argument("--dump-file", type=str, default=None, help="Output file path for sample dump (raw)")
    p.add_argument("--dump-format", choices=["f32le","f16le","i16le","i24le","i32le"], default="f32le", help="Raw format for dumped samples")
    p.add_argument("--dump-raw", action="store_true", help="Write RAW VBAN payload bytes (post-header) with no conversion")
    p.add_argument("--wav-path", type=str, default=None, help="Mirror the played stream to a WAV file (written in device output dtype)")
    p.add_argument("--wav-dtype", choices=["float32","int16","int32"], default="float32", help="WAV data type to write (matches device path)")
    p.add_argument("--device-blocksize", type=int, default=256, help="Assembled block size (frames) fed to audio device to reduce jitter")
    p.add_argument("--jitter-ms", type=float, default=10.0, help="Initial jitter buffer in milliseconds before starting playback")
    p.add_argument("--starve-fill", action="store_true", help="If playback starves, insert a block of silence to maintain clock")
    p.add_argument("--list-devices", action="store_true", help="List available output devices and exit")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--gain", type=float, default=1.0, help="Output gain multiplier (applied post-decode)")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        exit_code = asyncio.run(run(args))
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as e:
        # Suppress transport cleanup errors that happen after shutdown
        if "transport" in str(e).lower() or "connection_lost" in str(e).lower():
            exit_code = 0
        else:
            exit_code = 1
            print(f"Error: {e}")
    sys.exit(exit_code)
