"""
| File: udp_h264_streamer.py
| Author: Nataliia Yurevych (nataliia.yurevych@foxfour.ai)
| Description: Implements H.264 UDP RTP Streamer for Isaac Sim
"""

import cv2
import numpy as np
import socket
import struct
import threading
import time
import subprocess
import os
import fcntl
from typing import List

class H264RTPStreamer:
    """
    Construct the H.264 UDP RTP streamer object that uses FFmpeg for H.264 encoding

    Args:
        host (str): A string with the host address. Defaults to "127.0.0.1".
        port (int): The number of the port. Defaults to 8081.
        fps (int): The number of FPS. Defaults to 30.
        width (int): The width of streamed video. Defaults to 640.
        height (int): The height of streamed video. Defaults to 480.
        bitrate (int): The bitrate in kbps. Defaults to 1000.
        debug_output (bool): Allows debug output. Defaults to False.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8081,
                 fps: int = 30, width: int = 640, height: int = 480, bitrate: int = 1000, debug_output: bool = False):
        """Initialize H.264 RTP streamer"""
        self.host = host
        self.port = port
        self.fps = fps
        self.width = width
        self.height = height
        self.bitrate = bitrate  # kbps

        # RTP parameters
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sequence_number = 0
        self.timestamp = 0
        self.ssrc = 0x12345678
        self.timestamp_increment = 90000 // fps  # 90kHz clock

        # RTP packet parameters
        self.max_packet_size = 1400
        self.rtp_header_size = 12
        self.max_payload_size = self.max_packet_size - self.rtp_header_size

        # Threading
        self.running = False
        self.frame_queue = []
        self.queue_lock = threading.Lock()
        self.stream_thread = None

        # FFmpeg process for H.264 encoding
        self.ffmpeg_process = None

        # Debugging
        self.debug_output = debug_output

        print(f"H.264 RTP Streamer: {width}x{height}@{fps}fps, {bitrate}kbps to {host}:{port}")

    def start_ffmpeg_encoder(self):
        """Start FFmpeg process for H.264 encoding"""
        # FFmpeg command
        cmd = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',  # Input from stdin

            # H.264 encoding parameters
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-profile:v', 'baseline',
            '-level', '3.1',

            # Bitrate control
            '-b:v', f'{self.bitrate}k',
            '-maxrate', f'{self.bitrate}k',
            '-bufsize', f'{self.bitrate//2}k',

            # GOP structure
            '-g', str(self.fps),  # I-frame every second
            '-keyint_min', str(self.fps//2), # Minimum I-frame interval
            '-sc_threshold', '0', # Disable scene change detection

            # Annex B format with start codes
            '-bsf:v', 'h264_mp4toannexb',

            # Color space
            '-pix_fmt', 'yuv420p',

            # No audio
            '-an',

            # Error logging
            '-loglevel', 'error', '-report',

            # Output format: raw H.264 with NAL units
            '-f', 'h264',
            '-'  # Output to stdout
        ]

        print(f"Starting FFmpeg encoder: {' '.join(cmd[:10])}...")

        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

        # Test if FFmpeg started successfully
        time.sleep(0.5)
        if self.ffmpeg_process.poll() is None:
            print("FFmpeg H.264 encoder started successfully")
            return True
        else:
            stderr_output = self.ffmpeg_process.stderr.read().decode()
            print(f"ATTENTION! FFmpeg failed to start: {stderr_output}")
            return False

    def create_rtp_header(self, marker: bool = False) -> bytes:
        """Create RTP header for H.264 (payload type 96, dynamic type)"""
        # RTP version 2 (standard), no padding (the packet contains only the payload data),
        # no extension (no extra extension header), no CSRC (single source)
        first_byte = 0x80

        # Marker bit and payload type (96 for dynamic H.264)
        second_byte = 96
        if marker:
            second_byte |= 0x80

        sequence = self.sequence_number & 0xFFFF
        timestamp = int(self.timestamp) & 0xFFFFFFFF
        ssrc = self.ssrc & 0xFFFFFFFF

        return struct.pack('!BBHII', first_byte, second_byte, sequence, timestamp, ssrc)

    def find_nal_units(self, data: bytes) -> List[bytes]:
        """Find NAL units in H.264 data using start code detection"""
        nal_units = []

        if not data:
            return nal_units

        # Find all start code positions first
        start_positions = []

        i = 0
        while i < len(data) - 3:
            # Look for 4-byte start code (0x00 0x00 0x00 0x01)
            if (data[i] == 0x00 and data[i+1] == 0x00 and
                    data[i+2] == 0x00 and data[i+3] == 0x01):
                start_positions.append(i + 4)
                i += 4
            # Look for 3-byte start code (0x00 0x00 0x01)
            elif (data[i] == 0x00 and data[i+1] == 0x00 and data[i+2] == 0x01):
                start_positions.append(i + 3)
                i += 3
            else:
                i += 1

        # Extract NAL units
        for i in range(len(start_positions)):
            start = start_positions[i]

            # End is either next start code or end of data
            if i + 1 < len(start_positions):
                end = start_positions[i + 1] - 3  # Back up to before next start code
                # Check if it is a 4-byte start code (back up 4 instead of 3)
                if (end >= 4 and data[end-1] == 0x00 and data[end-2] == 0x00 and
                        data[end-3] == 0x00 and data[end-4] == 0x01):
                    end = start_positions[i + 1] - 4
            else:
                end = len(data)

            # Extract NAL unit
            if start < end:
                nal_unit = data[start:end]
                if len(nal_unit) > 0:
                    nal_units.append(nal_unit)

        return nal_units

    def fragment_nal_unit(self, nal_unit: bytes) -> List[bytes]:
        """Fragment large NAL units using FU-A fragmentation"""
        # if len(nal_unit) <= self.max_payload_size:
        #     # Single NAL unit packet
        #     return [nal_unit]

        # FU-A fragmentation needed
        packets = []

        # Get NAL unit header info
        nal_header = nal_unit[0]
        nal_type = nal_header & 0x1F
        nri = (nal_header >> 5) & 0x03

        # FU indicator: F=0, NRI=from original, Type=28 (FU-A)
        fu_indicator = (nri << 5) | 28

        # Fragment the payload (skip NAL header)
        payload = nal_unit[1:]
        offset = 0

        while offset < len(payload):
            # Calculate fragment size
            remaining = len(payload) - offset
            fragment_size = min(remaining, self.max_payload_size - 2)  # -2 for FU headers

            # FU header
            fu_header = nal_type
            if offset == 0:
                fu_header |= 0x80  # Start bit
            if offset + fragment_size >= len(payload):
                fu_header |= 0x40  # End bit

            # Create fragment
            fragment = bytes([fu_indicator, fu_header]) + payload[offset:offset + fragment_size]
            packets.append(fragment)

            offset += fragment_size

        return packets

    def send_nal_unit(self, nal_unit: bytes, is_last_nal: bool = False):
        """Send NAL unit via RTP with fragmentation"""
        if not nal_unit or len(nal_unit) < 1:
            return

        # Check NAL unit type
        nal_type = nal_unit[0] & 0x1F

        # Log I-frame transmission
        if self.debug_output:
            if nal_type != 1:  # I-frame
                print(f"Sending I-frame: {len(nal_unit)} bytes. Type: {nal_type}")
            elif nal_type == 1:  # P-frame
                print(f"Sending P-frame: {len(nal_unit)} bytes")

        # Show first few bytes of each NAL unit
        if self.debug_output:
            header_bytes = ' '.join([f'{b:02x}' for b in nal_unit[:8]])
            nal_names = {1: 'P-slice', 5: 'IDR-slice', 7: 'SPS', 8: 'PPS'}
            nal_name = nal_names.get(nal_type, f'Unknown-{nal_type}')
            print(f"NAL: {nal_name} ({nal_type}), {len(nal_unit)} bytes, header: {header_bytes}")

        # Fragment
        fragments = self.fragment_nal_unit(nal_unit)

        for i, fragment in enumerate(fragments):
            is_last_fragment = (i == len(fragments) - 1)
            marker = is_last_nal and is_last_fragment

            # Create and send RTP packet
            rtp_header = self.create_rtp_header(marker=marker)
            packet = rtp_header + fragment

            try:
                self.socket.sendto(packet, (self.host, self.port))
                self.sequence_number += 1
            except Exception as e:
                if e.errno == 9:  # Bad file descriptor after a reset/teardown
                    # Recreate the socket and skip this packet
                    print("UDP socket was closed; recreating...")
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    return
                else:
                    raise

            # Small delay between fragments
            if not is_last_fragment:
                time.sleep(0.0001)

    def is_ffmpeg_healthy(self):
        """Check if FFmpeg is still running and responsive"""
        if not self.ffmpeg_process or self.ffmpeg_process.poll() is not None:
            return False

        # Check if stderr has error messages
        try:
            # Non-blocking read of stderr
            fd = self.ffmpeg_process.stderr.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            stderr_data = self.ffmpeg_process.stderr.read(1024)
            if stderr_data and b'error' in stderr_data.lower():
                print(f"FFmpeg error detected: {stderr_data.decode()}")
                return False
        except:
            pass

        return True

    def encode_and_send_frame(self, frame: np.ndarray):
        """Encode a single camera frame to H.264 and send via RTP over UDP"""
        # Check health before processing
        if not self.is_ffmpeg_healthy():
            print("FFmpeg unhealthy before processing, restarting...")
            self.stop_ffmpeg_encoder()
            if not self.start_ffmpeg_encoder():
                return

        # Health check
        if not self.ffmpeg_process or self.ffmpeg_process.poll() is not None:
            print("FFmpeg process died, restarting...")
            if not self.start_ffmpeg_encoder():
                return

        # Make sure the frame matches the resolution expected by FFmpeg and resize frame if needed
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        # Send frame to FFmpeg
        frame_bytes = frame.tobytes() # Converts the image to raw bytes (in row-major BGR format)
        self.ffmpeg_process.stdin.write(frame_bytes) # Writes it to FFmpeg`s stdin
        self.ffmpeg_process.stdin.flush()

        # Make stdout non-blocking
        fd = self.ffmpeg_process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        # Read available data
        h264_data = b'' # Will accumulate the output H.264 byte stream
        max_attempts = 50 # To avoid infinite loops
        attempt = 0
        no_data_count = 0
        max_no_data_count = 15

        # Give FFmpeg initial time to start encoding
        time.sleep(0.005)

        while attempt < max_attempts and no_data_count < max_no_data_count:
            try:
                chunk = self.ffmpeg_process.stdout.read(8192) # Try to read 8192 bytes from FFmpeg’s stdout
                if chunk:
                    h264_data += chunk
                    no_data_count = 0  # Reset counter on successful read
                else:
                    no_data_count += 1
                    time.sleep(0.002)  # Wait for more data on no data
            except BlockingIOError:
                no_data_count += 1
                time.sleep(0.002)  # Wait before retry on blocking
            attempt += 1


        # Get NAL units
        nal_units = self.find_nal_units(h264_data)

        if self.debug_output:
            print("Attemting to send NAL units.")

        if nal_units:
            # Send each NAL unit
            for i, nal_unit in enumerate(nal_units):
                is_last = (i == len(nal_units) - 1)
                self.send_nal_unit(nal_unit, is_last)

            if self.debug_output:
                print(f"Sent {len(nal_units)} NAL units ({len(h264_data)} bytes)")
                print(f"No data count: {no_data_count}/{max_no_data_count}")
                print(f"H.264 data of length {len(h264_data)}: {h264_data[:50]}...")
        else:
           if self.debug_output:
                print("No valid NAL units found in H.264 data")
                print(f"No data count: {no_data_count}/{max_no_data_count}")
                print(f"H.264 data of length {len(h264_data)}: {h264_data}")

        self.timestamp += self.timestamp_increment

        # Check health after processing
        if len(h264_data) == 0 and no_data_count >= max_no_data_count:
            print(f"\n{'='*60}")
            print("FFMPEG STOPPED PRODUCING DATA, RESTARTING...")
            print(f"{'='*60}т")
            self.stop_ffmpeg_encoder()
            time.sleep(0.1)
            self.start_ffmpeg_encoder()

    def stop_ffmpeg_encoder(self):
        """Stop FFmpeg encoder process"""
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except:
                try:
                    self.ffmpeg_process.kill()
                    self.ffmpeg_process.wait(timeout=1)
                except:
                    pass
            self.ffmpeg_process = None

    def add_frame(self, frame: np.ndarray):
        """Add frame to processing queue"""
        with self.queue_lock:
            self.frame_queue.clear()  # Keep only latest frame
            self.frame_queue.append(frame.copy())

    def streaming_thread(self):
        """Background streaming thread"""
        frame_interval = 1.0 / self.fps

        while self.running:
            start_time = time.time()

            frame = None
            with self.queue_lock:
                if self.frame_queue:
                    frame = self.frame_queue.pop(0)
                    if self.debug_output:
                        print("Frame queue is not empty.")
                elif self.debug_output:
                    print("Frame queue is empty.")

            if frame is not None:
                self.encode_and_send_frame(frame)
                if self.debug_output:
                    print("Frame is not None.")
            elif self.debug_output:
                if self.debug_output:
                    print("Frame is None.")

            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self):
        """Start H.264 streaming"""
        if not self.start_ffmpeg_encoder():
            return False

        self.running = True
        self.stream_thread = threading.Thread(target=self.streaming_thread, daemon=True)
        self.stream_thread.start()

        print(f"Started H.264 RTP streaming to {self.host}:{self.port}")

        return True

    def stop(self):
        """Stop H.264 streaming"""
        self.running = False

        self.stream_thread.join(timeout=2.0)

        self.stop_ffmpeg_encoder()

        try:
            self.socket.close()
        except:
            pass

        print("Stopped H.264 RTP streaming")
