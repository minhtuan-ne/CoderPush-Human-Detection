import os
import time
import threading
import subprocess
import cv2
from pathlib import Path
import logging

class StreamManager:    
    def __init__(self, stream_url, output_file="live.ts", max_wait_time=30):
        self.stream_url = stream_url
        self.output_file = output_file
        self.max_wait_time = max_wait_time
        
        self.process = None
        self.is_ready = False
        self.error = None
        self.lock = threading.Lock()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Youtube cookies
        self.HSID = os.getenv('HSID')
        self.SSID = os.getenv('SSID')
        self.APISID = os.getenv('APISID')
        self.SAPISID = os.getenv('SAPISID')
        self.SID = os.getenv('SID')
        self.LOGIN_INFO = os.getenv('LOGIN_INFO')
        self.CONSISTENCY = os.getenv('CONSISTENCY')
        self.SIDCC = os.getenv('SIDCC')
        self.VISITOR_INFO1_LIVE = os.getenv('VISITOR_INFO1_LIVE')
        self.VISITOR_PRIVACY_METADATA = os.getenv('VISITOR_PRIVACY_METADATA')
    
    def init_stream(self):
        """Initialize the video stream and wait for it to be ready."""
        with self.lock:
            try:
                self._cleanup_previous()
                
                self.logger.info(f"Starting stream from: {self.stream_url}")
                self.process = subprocess.Popen(
                    ["streamlink",
                    "--force", 

                    "--http-cookie", f"HSID={self.HSID}",
                    "--http-cookie", f"SSID={self.SSID}",
                    "--http-cookie", f"APISID=o{self.APISID}",
                    "--http-cookie", f"SAPISID={self.SAPISID}",
                    "--http-cookie", f"SID={self.SID}",
                    "--http-cookie", f"LOGIN_INFO={self.LOGIN_INFO}",
                    "--http-cookie", f"CONSISTENCY={self.CONSISTENCY}",
                    "--http-cookie", f"SIDCC={self.SIDCC}",
                    "--http-cookie", f"VISITOR_INFO1_LIVE={self.VISITOR_INFO1_LIVE}",
                    "--http-cookie", f"VISITOR_PRIVACY_METADATA={self.VISITOR_PRIVACY_METADATA}",

                    self.stream_url,
                    "best", "-o", self.output_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                if self._wait_for_stream():
                    self.is_ready = True
                    self.error = None
                    self.logger.info("Stream initialized successfully!")
                    return True
                else:
                    self.error = f"Stream failed to initialize within {self.max_wait_time} seconds"
                    self.logger.error(self.error)
                    return False
                    
            except Exception as e:
                self.error = f"Error initializing stream: {str(e)}"
                self.logger.error(self.error)
                return False
    
    def _cleanup_previous(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        
        if os.path.exists(self.output_file):
            try:
                os.remove(self.output_file)
            except OSError as e:
                self.logger.warning(f"Could not remove {self.output_file}: {e}")
        
        self.is_ready = False
    
    def _wait_for_stream(self):
        start_time = time.time()
        
        while time.time() - start_time < self.max_wait_time:
            if self.process.poll() is not None:
                self.logger.error("Stream process terminated unexpectedly")
                return False
            
            if os.path.exists(self.output_file):
                file_size = os.path.getsize(self.output_file)
                if file_size > 1024:  # At least 1KB
                    if self._validate_video_file():
                        return True
            
            time.sleep(1)
        
        return False
    
    def _validate_video_file(self):
        """Validate that the video file can be opened and read."""
        try:
            cap = cv2.VideoCapture(self.output_file)
            if not cap.isOpened():
                return False
            
            ret, frame = cap.read()
            cap.release()
            
            return ret and frame is not None
        except Exception as e:
            self.logger.warning(f"Video validation failed: {e}")
            return False
    
    def is_healthy(self):
        """Check if the stream is still healthy."""
        with self.lock:
            if not self.is_ready:
                return False
            
            if self.process is None or self.process.poll() is not None:
                self.is_ready = False
                return False
            
            if not os.path.exists(self.output_file):
                self.is_ready = False
                return False
            
            return True
    
    def restart(self):
        """Restart the stream."""
        self.logger.info("Restarting stream...")
        return self.init_stream()
    
    def get_status(self):
        """Get current stream status."""
        return {
            "is_ready": self.is_ready,
            "is_healthy": self.is_healthy(),
            "error": self.error,
            "output_file_exists": os.path.exists(self.output_file),
            "output_file_size": os.path.getsize(self.output_file) if os.path.exists(self.output_file) else 0,
            "process_running": self.process is not None and self.process.poll() is None
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up stream resources...")
        with self.lock:
            self._cleanup_previous()
    
    def init_async(self):
        """Initialize stream in a background thread."""
        def _init():
            self.init_stream()
        
        thread = threading.Thread(target=_init, daemon=True)
        thread.start()
        return thread