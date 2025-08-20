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
from collections import deque
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class PacketBuffer:
    """High-performance circular buffer for storing packets"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.dropped_packets = 0
        self.total_packets = 0
        
        # Pre-allocate packet info objects for reuse
        self._packet_pool = []
        self._pool_size = min(1000, max_size // 10)  # Pool size based on buffer size
        
    def add_packet(self, packet_data: bytes, addr: tuple, timestamp: float) -> bool:
        """Add a packet to the buffer, returns True if added, False if dropped"""
        with self.lock:
            if len(self.buffer) >= self.buffer.maxlen:
                self.dropped_packets += 1
                return False
            
            # Reuse packet info object if available
            if self._packet_pool:
                packet_info = self._packet_pool.pop()
                packet_info['data'] = packet_data
                packet_info['addr'] = addr
                packet_info['timestamp'] = timestamp
            else:
                packet_info = {
                    'data': packet_data,
                    'addr': addr,
                    'timestamp': timestamp
                }
            
            self.buffer.append(packet_info)
            self.total_packets += 1
            return True
    
    def get_packet(self) -> Optional[dict]:
        """Get a packet from the buffer, returns None if empty"""
        with self.lock:
            if not self.buffer:
                return None
            
            packet_info = self.buffer.popleft()
            
            # Return packet info object to pool for reuse
            if len(self._packet_pool) < self._pool_size:
                self._packet_pool.append(packet_info)
            
            return packet_info
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        with self.lock:
            return {
                'current_size': len(self.buffer),
                'max_size': self.buffer.maxlen,
                'dropped_packets': self.dropped_packets,
                'total_packets': self.total_packets
            }

class UDPTrafficMonitor:
    def __init__(self, host: str, port: int, verbose: bool = False, high_performance: bool = False):
        self.host = host
        self.port = port
        self.verbose = verbose
        self.running = False
        self.packet_count = 0
        self.byte_count = 0
        self.start_time = None
        
        # VBAN packet tracking - PROPER duplicate detection
        self.vban_packets = {}  # stream_name -> last_packet_number
        self.expected_packet_numbers = {}  # stream_name -> next_expected_packet_number
        self.received_packet_numbers = {}  # stream_name -> set of received packet numbers
        self.duplicate_packets = 0
        self.lost_packets = 0
        self.total_vban_packets = 0
        
        # Time window for packet loss detection (sliding window) - DEPRECATED
        self.packet_window_seconds = 1.0
        self.packet_windows = {}  # stream_name -> deque of (packet_number, timestamp) pairs
        
        # PROPER sequential loss tracking
        self.stream_loss_tracking = {}  # stream_name -> {last_expected, total_lost}
        
        # Logging control
        self.last_stats_time = 0  # Track last stats display time
        self.stats_time_interval = 2.0  # Show stats every 2 seconds
        self.last_duplicate_log = 0  # Track last duplicate log time
        self.last_loss_log = 0       # Track last loss log time
        self.duplicate_log_interval = 5  # Log duplicates every N seconds
        self.loss_log_interval = 3       # Log losses every N seconds
        
        # Packet buffer for high-throughput processing
        if high_performance:
            buffer_size = 200000  # High-performance mode
        elif verbose:
            buffer_size = 50000   # Verbose mode (smaller buffer)
        else:
            buffer_size = 100000  # Default mode
        
        self.packet_buffer = PacketBuffer(buffer_size)
        self.analysis_running = False
        self.analysis_thread = None
        
        # Recent activity tracking for periodic summaries
        self.last_period_packet_count = 0
        self.last_period_byte_count = 0
        self.last_period_vban_count = 0
        
        # Rolling window for bitrate calculation (like main app)
        self.bitrate_window_seconds = 5.0
        self.bitrate_bytes_window = deque(maxlen=1000)  # Store (timestamp, bytes) pairs
        self.bitrate_packets_window = deque(maxlen=1000)  # Store (timestamp, packet_count) pairs
        self.last_bitrate_update = 0.0
        self.current_bitrate_bps = 0.0
        self.current_packet_rate = 0.0
        
        # Pre-allocate strings for better performance
        self._stats_template = "📊 Last {:.0f}s: {} packets ({:.1f} pkt/s), {} | Buffer: {}/{}"
        self._vban_template = " | VBAN: {} packets ({:.1f} pkt/s)"
        self._issue_template = " | Total: 🔄 {} dupes, ❌ {} lost"
        self._buffer_warning = " (⚠️ {} dropped)"
    
    def _calculate_current_lost_packets(self) -> int:
        """Calculate current lost packets using proper sequential tracking"""
        total_lost = 0
        for stream_name, tracking in self.stream_loss_tracking.items():
            total_lost += tracking.get('total_lost', 0)
        return total_lost
    
    def _track_packet_loss(self, stream_name: str, packet_number: int):
        """Track packet loss using sequential numbering with proper 32-bit wraparound handling"""
        if stream_name not in self.stream_loss_tracking:
            # First packet for this stream
            self.stream_loss_tracking[stream_name] = {
                'last_expected': packet_number + 1,
                'total_lost': 0,
                'last_packet': packet_number
            }
            return
        
        tracking = self.stream_loss_tracking[stream_name]
        last_expected = tracking['last_expected']
        last_packet = tracking['last_packet']
        
        # Handle 32-bit wraparound (0xFFFFFFFF -> 0)
        if packet_number < last_packet and last_packet > 0x7FFFFFFF:
            # Packet number wrapped around - this is normal, not a loss
            self._reset_stream_tracking(stream_name, packet_number)
            return
        
        if packet_number >= last_expected:
            # Packet is ahead of expected - some packets were lost
            lost_count = packet_number - last_expected
            if lost_count > 0:
                # Sanity check: don't count unrealistic losses
                if lost_count < 1000000:  # Max 1M packets lost at once
                    tracking['total_lost'] += lost_count
                    if self.verbose:
                        logger.warning(f"❌ LOST: Stream '{stream_name}' lost {lost_count} packets before {packet_number}")
                else:
                    # Unrealistic loss count - likely wraparound issue
                    if self.verbose:
                        logger.warning(f"⚠️  Wraparound detected: packet {packet_number} after {last_expected} (gap: {lost_count})")
            # Update expected to this packet + 1
            tracking['last_expected'] = packet_number + 1
        elif packet_number < last_expected - 1:
            # Packet is behind expected - out of order delivery (not a loss)
            # Only update expected if this packet is newer than what we had
            if packet_number >= tracking['last_expected'] - 1:
                tracking['last_expected'] = packet_number + 1
        else:
            # packet_number == last_expected - 1 (normal sequential)
            tracking['last_expected'] = packet_number + 1
        
        tracking['last_packet'] = packet_number
    
    def _reset_stream_tracking(self, stream_name: str, packet_number: int):
        """Reset tracking for a stream when wraparound is detected"""
        if self.verbose:
            logger.info(f"🔄 Resetting tracking for stream '{stream_name}' at packet {packet_number}")
        
        # Reset loss tracking
        if stream_name in self.stream_loss_tracking:
            self.stream_loss_tracking[stream_name] = {
                'last_expected': packet_number + 1,
                'total_lost': 0,
                'last_packet': packet_number
            }
        
        # Reset expected packet numbers
        if stream_name in self.expected_packet_numbers:
            self.expected_packet_numbers[stream_name] = packet_number + 1
        
        # Clear received packet numbers (start fresh)
        if stream_name in self.received_packet_numbers:
            self.received_packet_numbers[stream_name].clear()
            self.received_packet_numbers[stream_name].add(packet_number)
    
    def _is_duplicate_packet(self, stream_name: str, packet_number: int) -> bool:
        """Properly detect if a packet is a true duplicate with wraparound handling"""
        if stream_name not in self.received_packet_numbers:
            self.received_packet_numbers[stream_name] = set()
        
        # Handle 32-bit wraparound - if we see a much smaller number after a large one,
        # it's likely a wraparound, not a duplicate
        if stream_name in self.vban_packets:
            last_packet = self.vban_packets[stream_name]
            if packet_number < last_packet and last_packet > 0x7FFFFFFF:
                # Likely wraparound - reset all tracking for this stream
                self._reset_stream_tracking(stream_name, packet_number)
                return False  # Not a duplicate after reset
        
        # A packet is a duplicate if we've already received this exact packet number
        # We need to track what we've received, not what we expect next
        if packet_number in self.received_packet_numbers[stream_name]:
            # We've already received this packet number - it's a duplicate
            return True
        
        # This is a new packet number - add it to received set
        self.received_packet_numbers[stream_name].add(packet_number)
        return False
    
    def _update_expected_packet_number(self, stream_name: str, packet_number: int):
        """Update the expected packet number for the next packet"""
        if stream_name not in self.expected_packet_numbers:
            # First packet for this stream - expect the next sequential number
            self.expected_packet_numbers[stream_name] = packet_number + 1
        else:
            # Update expected to next sequential number
            current_expected = self.expected_packet_numbers[stream_name]
            
            # Handle normal sequential case
            if packet_number == current_expected - 1:
                # This is the expected packet, move to next
                self.expected_packet_numbers[stream_name] = packet_number + 1
            elif packet_number > current_expected - 1:
                # Packet is ahead of expected (some packets were lost)
                # Update expected to this packet + 1
                self.expected_packet_numbers[stream_name] = packet_number + 1
            # If packet_number < current_expected - 1, it's out of order (not a duplicate)
    
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
            
            # Ensure packet buffer is ready before starting analysis thread
            if not hasattr(self, 'packet_buffer'):
                logger.error("❌ Packet buffer not initialized!")
                return
            
            # Start analysis thread
            self.analysis_running = True
            self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
            self.analysis_thread.start()
            
            # Monitor for packets
            while self.running:
                try:
                    # Try to receive data (non-blocking)
                    data, addr = sock.recvfrom(65536)  # Max UDP packet size
                    
                    # Add packet to buffer (non-blocking) - with safety check
                    timestamp = time.time()
                    if hasattr(self, 'packet_buffer'):
                        try:
                            if not self.packet_buffer.add_packet(data, addr, timestamp):
                                # Buffer full - this can happen with very high bandwidth
                                if self.packet_count % 1000 == 0:  # Log occasionally
                                    logger.warning("⚠️  Packet buffer full - some packets may be dropped")
                        except Exception as e:
                            logger.error(f"Error adding packet to buffer: {e}")
                    else:
                        # Packet buffer not available, just count the packet
                        self.packet_count += 1
                        self.byte_count += len(data)
                    
                except BlockingIOError:
                    # No data available, check if we should stop
                    await asyncio.sleep(0.001)  # Reduced sleep for higher responsiveness
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
    
    def _process_packet(self, data: bytes, addr: tuple, timestamp: float):
        """Process a received UDP packet"""
        self.packet_count += 1
        self.byte_count += len(data)
        
        # Add current packet to rolling window for bitrate calculation
        current_time = time.time()
        self.bitrate_bytes_window.append((current_time, len(data)))
        self.bitrate_packets_window.append((current_time, 1))
        
        # Update bitrate every 500ms (like main app)
        if current_time - self.last_bitrate_update >= 0.5:
            # Remove old entries outside the window
            cutoff_time = current_time - self.bitrate_window_seconds
            while self.bitrate_bytes_window and self.bitrate_bytes_window[0][0] < cutoff_time:
                self.bitrate_bytes_window.popleft()
            while self.bitrate_packets_window and self.bitrate_packets_window[0][0] < cutoff_time:
                self.bitrate_packets_window.popleft()
            
            # Calculate bitrate from window data (exactly like main app)
            if self.bitrate_bytes_window:
                window_bytes = sum(bytes_data for _, bytes_data in self.bitrate_bytes_window)
                window_duration = max(0.1, current_time - self.bitrate_bytes_window[0][0])
                self.current_bitrate_bps = (window_bytes * 8) / window_duration
                
                # Calculate packet rate from window
                if self.bitrate_packets_window:
                    window_packets = sum(packet_count for _, packet_count in self.bitrate_packets_window)
                    self.current_packet_rate = window_packets / window_duration
            else:
                self.current_bitrate_bps = 0.0
                self.current_packet_rate = 0.0
            
            self.last_bitrate_update = current_time
        
        # Basic packet analysis
        timestamp_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]
        src_ip, src_port = addr
        packet_size = len(data)
        
        # Check if it looks like VBAN
        is_vban = len(data) >= 4 and data[:4] == b'VBAN'
        
        # VBAN packet analysis
        if is_vban and len(data) >= 32:
            self._analyze_vban_packet(data, timestamp, src_ip, src_port)
        
        # Show periodic summary every 2 seconds instead of per-packet info
        if (current_time - self.last_stats_time) >= self.stats_time_interval:
            self._show_stats()
            self.last_stats_time = current_time
    
    def _analysis_worker(self):
        """Worker thread that processes packets from the buffer"""
        while self.analysis_running:
            try:
                packet_info = self.packet_buffer.get_packet()
                if packet_info:
                    # Process the packet
                    self._process_packet(packet_info['data'], packet_info['addr'], packet_info['timestamp'])
                else:
                    # No packets in buffer, sleep briefly
                    time.sleep(0.001)  # 1ms sleep for responsiveness
            except Exception as e:
                # Log any errors but continue
                logger.error(f"Error in analysis worker: {e}")
                time.sleep(0.001)
                continue
    
    def _analyze_vban_packet(self, data: bytes, timestamp: float, src_ip: str, src_port: int):
        """Analyze VBAN packet for duplicates and packet loss using proper sequential tracking"""
        try:
            # Parse VBAN header fields
            stream_name = data[4:20].decode('utf-8').rstrip('\x00')
            packet_number = int.from_bytes(data[28:32], 'little')  # nuFrame field
            current_time = time.time()
            
            self.total_vban_packets += 1
            
            # Initialize packet window for this stream if needed (for loss detection)
            if stream_name not in self.packet_windows:
                self.packet_windows[stream_name] = deque(maxlen=1000)  # Max 1000 packets per stream
                if self.verbose:
                    logger.info(f"🆕 NEW STREAM: '{stream_name}' starting at packet {packet_number}")
            
            # PROPER DUPLICATE DETECTION
            is_duplicate = self._is_duplicate_packet(stream_name, packet_number)
            if is_duplicate:
                self.duplicate_packets += 1
                # Log duplicates at controlled intervals to prevent spam
                if self.verbose and (current_time - self.last_duplicate_log) >= self.duplicate_log_interval:
                    logger.warning(f"🔄 DUPLICATE: Stream '{stream_name}' packet {packet_number} (duplicate #{self.duplicate_packets})")
                    self.last_duplicate_log = current_time
            
            # Update expected packet number for next packet
            self._update_expected_packet_number(stream_name, packet_number)
            
            # Track packet loss using proper sequential method
            self._track_packet_loss(stream_name, packet_number)
            
            # Keep track of last packet for stream details display
            self.vban_packets[stream_name] = max(self.vban_packets.get(stream_name, 0), packet_number)
            
        except Exception as e:
            if self.verbose:
                logger.error(f"⚠️  Error analyzing VBAN packet: {e}")
    
    def _show_stats(self):
        """Show current statistics"""
        if self.start_time:
            elapsed = time.time() - self.start_time.timestamp()
            
            # Calculate activity since last period
            period_packets = self.packet_count - self.last_period_packet_count
            period_bytes = self.byte_count - self.last_period_byte_count
            period_vban = self.total_vban_packets - self.last_period_vban_count
            
            period_pps = period_packets / self.stats_time_interval
            period_vban_pps = period_vban / self.stats_time_interval
            
            # Use rolling window bitrate (like main app) instead of period-based
            if self.current_bitrate_bps > 1_000_000:
                bps_str = f"{self.current_bitrate_bps/1_000_000:.1f} Mbps"
            elif self.current_bitrate_bps > 1_000:
                bps_str = f"{self.current_bitrate_bps/1_000:.1f} Kbps"
            else:
                bps_str = f"{self.current_bitrate_bps:.0f} bps"
            
            # Get buffer stats once (with safety check)
            if hasattr(self, 'packet_buffer'):
                try:
                    buffer_stats = self.packet_buffer.get_stats()
                    
                    # Use pre-allocated templates for better performance
                    stats_msg = self._stats_template.format(
                        self.stats_time_interval, period_packets, period_pps, bps_str,
                        buffer_stats['current_size'], buffer_stats['max_size']
                    )
                    
                    # Add current packet rate info
                    if self.current_packet_rate > 0:
                        stats_msg += f" | Current: {self.current_packet_rate:.1f} pkt/s"
                    
                    # Add buffer warning if needed
                    if buffer_stats['dropped_packets'] > 0:
                        stats_msg += self._buffer_warning.format(buffer_stats['dropped_packets'])
                except Exception as e:
                    # Fallback if buffer stats fail
                    stats_msg = f"📊 Last {self.stats_time_interval:.0f}s: {period_packets} packets ({period_pps:.1f} pkt/s), {bps_str} | Buffer: unavailable"
            else:
                # Fallback if no packet buffer
                stats_msg = f"📊 Last {self.stats_time_interval:.0f}s: {period_packets} packets ({period_pps:.1f} pkt/s), {bps_str} | Buffer: unavailable"
            
            # Add VBAN stats if available
            if period_vban > 0:
                stats_msg += self._vban_template.format(period_vban, period_vban_pps)
                
                # Add issue summary if any
                current_lost = self._calculate_current_lost_packets()
                if self.duplicate_packets > 0 or current_lost > 0:
                    stats_msg += self._issue_template.format(self.duplicate_packets, current_lost)
            
            logger.info(stats_msg)
            
            # Update last period counters
            self.last_period_packet_count = self.packet_count
            self.last_period_byte_count = self.byte_count
            self.last_period_vban_count = self.total_vban_packets
    
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
            
            # Buffer summary (with safety check)
            if hasattr(self, 'packet_buffer'):
                try:
                    buffer_stats = self.packet_buffer.get_stats()
                    logger.info(f"📋 Buffer Summary: {buffer_stats['total_packets']} processed, {buffer_stats['dropped_packets']} dropped")
                    if buffer_stats['dropped_packets'] > 0:
                        drop_rate = (buffer_stats['dropped_packets'] / buffer_stats['total_packets']) * 100
                        logger.warning(f"⚠️  Buffer drop rate: {drop_rate:.2f}% - consider increasing buffer size for high bandwidth")
                except Exception as e:
                    logger.warning(f"⚠️  Could not get buffer stats: {e}")
            else:
                logger.warning("⚠️  Packet buffer not available for summary")
            
            # VBAN analysis summary
            if self.total_vban_packets > 0:
                logger.info("-" * 40)
                logger.info("🎵 VBAN ANALYSIS SUMMARY")
                logger.info(f"📻 Streams detected: {len(self.vban_packets)}")
                logger.info(f"🔢 Total VBAN packets: {self.total_vban_packets}")
                logger.info(f"🔄 Duplicate packets: {self.duplicate_packets}")
                
                # Show current lost packets from time windows
                current_lost = self._calculate_current_lost_packets()
                logger.info(f"❌ Current lost packets: {current_lost} (in {self.packet_window_seconds}s window)")
                
                # Show current bitrate from rolling window
                if self.current_bitrate_bps > 0:
                    if self.current_bitrate_bps > 1_000_000:
                        bitrate_str = f"{self.current_bitrate_bps/1_000_000:.2f} Mbps"
                    elif self.current_bitrate_bps > 1_000:
                        bitrate_str = f"{self.current_bitrate_bps/1_000:.2f} Kbps"
                    else:
                        bitrate_str = f"{self.current_bitrate_bps:.0f} bps"
                    logger.info(f"📊 Current bitrate: {bitrate_str}")
                
                current_lost = self._calculate_current_lost_packets()
                if self.duplicate_packets > 0 or current_lost > 0:
                    # CORRECT calculation: duplicates as % of unique packets expected
                    unique_packets_expected = max(1, self.total_vban_packets - self.duplicate_packets)
                    duplicate_rate = (self.duplicate_packets / unique_packets_expected) * 100
                    
                    # Correct calculation: lost packets as % of total expected packets
                    total_expected_packets = self.total_vban_packets + current_lost
                    loss_rate = (current_lost / total_expected_packets) * 100
                    
                    logger.warning(f"⚠️  Duplicate rate: {duplicate_rate:.1f}%")
                    logger.warning(f"⚠️  Loss rate: {loss_rate:.1f}%")
                    
                    if duplicate_rate > 5:
                        logger.warning("💡 High duplicate rate suggests network congestion or retransmission issues")
                    if loss_rate > 2:
                        logger.warning("💡 High loss rate suggests network instability or bandwidth issues")
                else:
                    logger.info("✅ Perfect packet delivery - no duplicates or losses detected!")
                
                # Show stream details
                for stream_name, last_packet in self.vban_packets.items():
                    logger.info(f"   📻 '{stream_name}': packets 0-{last_packet}")
            
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
        self.analysis_running = False
        
        # Wait for analysis thread to finish
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)

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
  %(prog)s -q                 # Quiet mode - minimal output
  %(prog)s --high-performance # Maximum performance mode for high bandwidth
  %(prog)s --packet-window 2.0 # 2-second window for packet loss detection
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
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode - minimal output, only errors and final summary'
    )
    
    parser.add_argument(
        '--high-performance',
        action='store_true',
        help='High-performance mode - optimized for maximum throughput monitoring'
    )
    
    parser.add_argument(
        '--packet-window',
        type=float,
        default=1.0,
        help='Time window in seconds for packet loss detection (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.port < 1 or args.port > 65535:
        logger.error("❌ Port must be between 1 and 65535")
        return 1
    
    # Create and start monitor
    monitor = UDPTrafficMonitor(args.host, args.port, args.verbose, args.high_performance)
    
    # Apply quiet mode settings
    if args.quiet:
        monitor.stats_time_interval = 10.0  # Show stats every 10 seconds in quiet mode
        monitor.duplicate_log_interval = 30  # Log duplicates every 30 seconds
        monitor.loss_log_interval = 20       # Log losses every 20 seconds
    
    # Apply high-performance optimizations
    if args.high_performance:
        monitor.stats_time_interval = 5.0   # Show stats every 5 seconds
        monitor.duplicate_log_interval = 60  # Log duplicates every 60 seconds
        monitor.loss_log_interval = 60       # Log losses every 60 seconds
        # Buffer size already set during initialization
    
    # Apply packet window setting
    monitor.packet_window_seconds = args.packet_window
    
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
