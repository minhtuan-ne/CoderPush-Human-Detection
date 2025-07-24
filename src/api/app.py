from flask_socketio import SocketIO, emit
from flask import Flask, request
from stream_manager import StreamManager
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from ec2.face_detector import FaceDetector

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
STREAM_URL = "https://www.youtube.com/watch?v=VR-x3HdhKLQ"
VIDEO_FILE = "live.ts"
MAX_FRAMES = 100

# Initialize components
stream_manager = StreamManager(STREAM_URL, output_file=VIDEO_FILE)
detector = FaceDetector()

@socketio.on("connect")
def handle_connect():
    print("Client connected")

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")

@socketio.on("start_processing")
def handle_start_processing():
    sid = request.sid
    print(f"Starting background processing loop for client {sid}")
    socketio.start_background_task(target=process_loop, sid=sid)

def process_loop(sid):
    while True:
        print(f"[{sid}] Initializing stream...")
        stream_manager.init_stream()

        # Wait for video file to be ready
        wait_time = 0
        while not os.path.exists(VIDEO_FILE) or os.path.getsize(VIDEO_FILE) < 1000:
            time.sleep(0.5)
            wait_time += 0.5
            if wait_time > 10:
                socketio.emit("processing_error", {"error": "Video file not ready after 10s"}, to=sid)
                return

        try:
            detector.process_video_stream(
                video_source=VIDEO_FILE,
                max_frames=MAX_FRAMES,
                emit_func=lambda result: socketio.emit("frame_processed", result, to=sid)
            )
            socketio.emit("processing_done", {"status": "completed"}, to=sid)
            time.sleep(1)  # Wait before restarting
        except Exception as e:
            socketio.emit("processing_error", {"error": str(e)}, to=sid)
            return

if __name__ == "__main__":
    socketio.run(app, debug=True, port=7860, host="0.0.0.0")
