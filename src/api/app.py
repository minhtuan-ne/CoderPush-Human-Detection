from flask_socketio import SocketIO, emit # type: ignore
from flask import Flask, request
from stream_manager import StreamManager
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from ec2.face_detector import FaceDetector

app = Flask(__name__)
socketio = SocketIO(app, 
                    cors_allowed_origins="*",
                    ping_timeout=60,
                    ping_interval=5)

# Configuration
STREAM_URL_1 = "https://www.youtube.com/watch?v=cH7VBI4QQzA"
STREAM_URL_2 = "https://www.youtube.com/watch?v=VR-x3HdhKLQ"
VIDEO_FILE = "live.ts"
MAX_FRAMES = 100

# Initialize components
stream_manager = StreamManager(STREAM_URL_1, output_file=VIDEO_FILE)
detector = FaceDetector()

@socketio.on("connect")
def handle_connect():
    print("Client connecting")
    socketio.emit("connected", "Client connected")
    socketio.start_background_task(process_frame)

@socketio.on("connected")
def handle_connected():
    print("Client connected")    

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")

# Handle manual "ping" event from client
@socketio.on('ping_from_client')
def handle_ping():
    # Respond with "pong" and current server time
    emit('pong_from_server', {'server_time': time.time()})

def process_frame():
    while True:
        stream_manager.init_stream()
        results = detector.process_video_stream(
                    video_source=VIDEO_FILE,
                    max_frames=MAX_FRAMES
        )
        if results:
            print(results)
            socketio.emit("frame_processed", results)
        
        socketio.sleep(1)


if __name__ == "__main__":
    socketio.run(app, port=7860, host="0.0.0.0")
