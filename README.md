# VBAN to BlackHole Audio Receiver

A Python application that receives VBAN (Virtual Broadcast Audio Network) audio over UDP and plays it through CoreAudio outputs like BlackHole. It decodes LPCM audio, provides real-time VU meters with legit ballistics, optional WAV mirroring, and raw payload dumps for analysis.

## Features

- **VBAN UDP Receiver**: Parses VBAN header and decodes LPCM payload (PCM 8/16/16u/24/32, float16/float32)
- **Interleaved Audio**: WAV-style interleaving as per VBAN spec
- **Multi-channel Support**: Configurable channel count
- **Real-time VU Meters**: ASCII VU meters with ~300 ms integration and separate peak decay. Peak marker overlaid on bar
- **CoreAudio Integration**: Direct output to BlackHole or any CoreAudio device
- **Gain Control**: Adjustable output gain
- **Jitter Buffer & Block Assembly**: Small configurable jitter buffer and fixed device blocksize to reduce dropouts
- **WAV Mirroring**: Mirror exactly what is sent to the device into a WAV file (float32/int16/int32)
- **Raw Dump**: Dump raw VBAN payload bytes or decoded samples for offline analysis

## Installation

### Prerequisites

- **Python 3.9+**
- **macOS** (for CoreAudio support)
- **BlackHole** (optional, for virtual audio routing)

### Install BlackHole (Optional)

If you want to route audio to other applications:

```bash
# Install via Homebrew
brew install blackhole-2ch

# Or download from: https://existential.audio/blackhole/
```

### Install Python Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd vban-to-blackhole

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

```bash
# List available audio devices
python3 vban_to_blackhole16.py --list-devices
```

## Usage

### Basic Usage

Receive any VBAN audio stream with default settings:

```bash
python3 vban_to_blackhole16.py
```

### Custom Configuration

```bash
# Listen on specific IP/port
python3 vban_to_blackhole16.py --listen-ip 192.168.1.100 --listen-port 6980

# Common 2-channel path to BlackHole 16ch
python3 vban_to_blackhole16.py \
  --in-channels 2 --input-format pcm16 --byte-order little \
  --channels 2 --sample-rate 48000 \
  --output-device "BlackHole 16ch"

# Adjust gain
python3 vban_to_blackhole16.py --gain 2.0  # Increase volume
python3 vban_to_blackhole16.py --gain 0.5  # Decrease volume

# Verbose logging (shows headers and events)
python3 vban_to_blackhole16.py --verbose
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--listen-ip` | `0.0.0.0` | IP address to bind for UDP VBAN packets |
| `--listen-port` | `6980` | UDP port to listen on |
| `--sample-rate` | `48000` | Output sample rate (Hz) |
| `--channels` | `2` | Number of output channels (to device/WAV) |
| `--output-device` | `BlackHole 16ch` | Output device name or index |
| `--in-channels` |  | Override input (sender) channel count |
| `--input-format` | `auto` | `auto`, `pcm8`, `pcm16`, `pcm16u`, `pcm24`, `pcm32`, `float16`, `float32` |
| `--byte-order` | `little` | `little` or `big` |
| `--auto-format` | `False` | Auto-detect format/byte order per packet |
| `--gain` | `1.0` | Output gain multiplier |
| `--device-blocksize` | `256` | Frames per block sent to device |
| `--jitter-ms` | `10.0` | Initial jitter buffer before starting playback |
| `--starve-fill` | `False` | Insert silence if playback starves |
| `--wav-path` |  | Mirror device stream to WAV file |
| `--wav-dtype` | `float32` | WAV dtype: `float32`, `int16`, `int32` |
| `--dump-seconds` |  | Duration to dump raw/decoded samples |
| `--dump-file` |  | Output file for dump |
| `--dump-format` | `f32le` | Decoded dump format (when not `--dump-raw`) |
| `--dump-raw` | `False` | Dump raw VBAN payload bytes (post-header) |

## Examples

### 2-Channel VBAN Receiver to BlackHole

```bash
python3 vban_to_blackhole16.py \
  --in-channels 2 --input-format pcm16 --byte-order little \
  --channels 2 --sample-rate 48000 \
  --output-device "BlackHole 16ch"
```

### Custom Network Configuration

```bash
python3 vban_to_blackhole16.py \
  --listen-ip 192.168.1.100 \
  --listen-port 6981 \
  --channels 2 \
  --sample-rate 96000 \
  --output-device "BlackHole 16ch"
```

## VU Meters

When not in verbose mode, the application displays real-time ASCII VU meters with legit ballistics (~300 ms integration) and separate peak decay. A thin marker shows peak within the bar.

## Troubleshooting

### No Audio Output

1. **Check device selection**:
   ```bash
   python3 vban_to_blackhole16.py --list-devices
   ```
2. **Verify BlackHole is running**:
   - Check Audio MIDI Setup app
   - Ensure BlackHole appears in output devices
3. **Check VBAN source**:
   - Verify sender is transmitting on correct IP/port
   - Confirm stream format matches expected channels

### High Latency or Dropouts

- Increase `--device-blocksize` (e.g., 1024 or 2048)
- Set a small `--jitter-ms` (10–40 ms) and enable `--starve-fill`
- Ensure `--input-format` and `--byte-order` match the sender

### Audio device not found

```bash
# List all output-capable devices
python3 vban_to_blackhole16.py --list-devices
```

## Development

### Project Structure

```
vban-to-blackhole/
├── vban_to_blackhole16.py  # Main application
├── vban_sample_analyzer.py # Optional analyzer tool
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

### Dependencies

- **sounddevice**: Audio I/O via PortAudio
- **soundfile**: WAV writing (libsndfile)
- **numpy**: Audio data processing

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Support

For issues and questions:
1. Check this README and troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed information

## Reference

This project’s goals were inspired by the asyncio VBAN player example: [`aiovban_pyaudio/player.py`](https://github.com/wmbest2/aiovban/blob/main/aiovban_pyaudio/src/aiovban_pyaudio/player.py).
