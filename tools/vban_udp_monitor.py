#!/usr/bin/env python3
"""
VBAN UDP Traffic Monitor

A simple tool to monitor UDP traffic on VBAN ports to help debug network connectivity.
This tool listens for ANY UDP packets (not just VBAN) to see if there's network traffic at all.
"""

import asyncio
import socket
import argparse
import logging
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class UDPTrafficMonitor:
    def __init__(self, host: str, port: int, verbose: bool = False):
        self.host = host
        self.port = port
        self.verbose = verbose
        self.running = False
        self.packet_count = 0
        self.byte_count = 0
        self.start_time = None
        
    async def start_monitoring(self):
        """Start monitoring UDP traffic on the specified port"""
        self.running = True
        self.start_time = datetime.now()
        
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            # Bind to the specified address and port
            sock.bind((self.host, self.port))
            logger.info(f"🔍 Monitoring UDP traffic on {self.host}:{self.port}")
            logger.info(f"📊 Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("⏹️  Press Ctrl+C to stop monitoring")
            logger.info("-" * 60)
            
            # Set socket to non-blocking mode
            sock.setblocking(False)
            
            # Monitor for packets
            while self.running:
                try:
                    # Try to receive data (non-blocking)
                    data, addr = sock.recvfrom(65536)  # Max UDP packet size
                    
                    # Process the packet
                    self._process_packet(data, addr)
                    
                except BlockingIOError:
                    # No data available, check if we should stop
                    await asyncio.sleep(0.1)
                    continue
                    
                except Exception as e:
                    logger.error(f"Error receiving packet: {e}")
                    continue
                    
        except OSError as e:
            if e.errno == 48:  # Address already in use
                logger.error(f"❌ Port {self.port} is already in use. Try a different port or stop other applications.")
                logger.info("💡 You can check what's using the port with: lsof -i :{self.port}")
            else:
                logger.error(f"❌ Failed to bind to {self.host}:{self.port}: {e}")
            return
            
        finally:
            sock.close()
            self._print_summary()
    
    def _process_packet(self, data: bytes, addr: tuple):
        """Process a received UDP packet"""
        self.packet_count += 1
        self.byte_count += len(data)
        
        # Basic packet analysis
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        src_ip, src_port = addr
        packet_size = len(data)
        
        # Check if it looks like VBAN
        is_vban = len(data) >= 4 and data[:4] == b'VBAN'
        
        # Display packet info
        if self.verbose:
            # Verbose mode - show packet details
            logger.info(f"📦 Packet #{self.packet_count} from {src_ip}:{src_port}")
            logger.info(f"   📏 Size: {packet_size} bytes")
            logger.info(f"   🕐 Time: {timestamp}")
            logger.info(f"   🏷️  VBAN: {'✅ Yes' if is_vban else '❌ No'}")
            
            if is_vban and len(data) >= 28:
                try:
                    # Try to parse VBAN header
                    stream_name = data[4:20].decode('utf-8').rstrip('\x00')
                    sample_rate = int.from_bytes(data[20:24], 'little')
                    sample_count = int.from_bytes(data[24:26], 'little')
                    channel_count = int.from_bytes(data[26:27], 'little')
                    data_type = int.from_bytes(data[27:28], 'little')
                    
                    logger.info(f"   📻 Stream: '{stream_name}'")
                    logger.info(f"   🎵 Sample Rate: {sample_rate} Hz")
                    logger.info(f"   🔊 Samples: {sample_count}")
                    logger.info(f"   🎧 Channels: {channel_count}")
                    logger.info(f"   🔢 Data Type: 0x{data_type:02x}")
                except Exception as e:
                    logger.info(f"   ⚠️  Could not parse VBAN header: {e}")
            
            # Show first few bytes as hex
            hex_data = ' '.join(f'{b:02x}' for b in data[:16])
            logger.info(f"   🔍 Data: {hex_data}{'...' if len(data) > 16 else ''}")
            logger.info("-" * 40)
        else:
            # Simple mode - just show basic info
            vban_indicator = "🔴" if is_vban else "⚪"
            logger.info(f"{vban_indicator} {timestamp} | {src_ip}:{src_port} | {packet_size:4d} bytes | {'VBAN' if is_vban else 'Other'}")
        
        # Show stats every 10 packets
        if self.packet_count % 10 == 0:
            self._show_stats()
    
    def _show_stats(self):
        """Show current statistics"""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            pps = self.packet_count / elapsed if elapsed > 0 else 0
            bps = self.byte_count / elapsed if elapsed > 0 else 0
            
            logger.info(f"📊 Stats: {self.packet_count} packets, {self.byte_count:,} bytes, {pps:.1f} pkt/s, {bps:.0f} B/s")
    
    def _print_summary(self):
        """Print final summary when monitoring stops"""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info("-" * 60)
            logger.info("📊 MONITORING STOPPED")
            logger.info(f"⏱️  Duration: {elapsed:.1f} seconds")
            logger.info(f"📦 Total Packets: {self.packet_count}")
            logger.info(f"💾 Total Bytes: {self.byte_count:,}")
            if elapsed > 0:
                logger.info(f"🚀 Average: {self.packet_count/elapsed:.1f} packets/sec, {self.byte_count/elapsed:.0f} bytes/sec")
            
            if self.packet_count == 0:
                logger.warning("⚠️  No packets received! Check:")
                logger.warning("   • Is the VBAN sender configured correctly?")
                logger.warning("   • Is the sender targeting the right IP address?")
                logger.warning("   • Are there any firewalls blocking UDP traffic?")
                logger.warning("   • Is the network routing correct?")
            else:
                logger.info("✅ Packets received successfully!")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False

async def main():
    parser = argparse.ArgumentParser(
        description="Monitor UDP traffic on VBAN ports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Monitor on 0.0.0.0:6980 (default VBAN port)
  %(prog)s -p 6981           # Monitor on port 6981
  %(prog)s -H 127.0.0.1      # Monitor only localhost
  %(prog)s -v                 # Verbose mode with packet details
  %(prog)s -H 0.0.0.0 -p 6980 -v  # Full verbose monitoring
        """
    )
    
    parser.add_argument(
        '-H', '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0 for all interfaces)'
    )
    
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=6980,
        help='Port to monitor (default: 6980, standard VBAN port)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose mode - show detailed packet information'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.port < 1 or args.port > 65535:
        logger.error("❌ Port must be between 1 and 65535")
        return 1
    
    # Create and start monitor
    monitor = UDPTrafficMonitor(args.host, args.port, args.verbose)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Received Ctrl+C, stopping monitor...")
        monitor.stop()
        # Give it a moment to finish
        await asyncio.sleep(0.1)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        exit(130)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        exit(1)
