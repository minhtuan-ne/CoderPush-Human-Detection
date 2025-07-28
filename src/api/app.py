from flask_socketio import SocketIO # type: ignore
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
                    ping_timeout=250,
                    ping_interval=25)

# Configuration
STREAM_URL_1 = "https://www.youtube.com/watch?v=cH7VBI4QQzA"
STREAM_URL_2 = "https://www.youtube.com/watch?v=VR-x3HdhKLQ"
VIDEO_FILE = "live.ts"
MAX_FRAMES = 100

# Initialize components
stream_manager = StreamManager(STREAM_URL_2, output_file=VIDEO_FILE)
detector = FaceDetector()

@socketio.on("connect")
def handle_connect():
    print("Client connecting")
    socketio.emit("connected", "Client connected")
    socketio.emit('pong_from_server', {'server_time': time.time()})
    socketio.start_background_task(process_frame)

@socketio.on("connected")
def handle_connected():
    print("Client connected")    

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")

def server_ping():
    while True:
        try:
            socketio.emit('ping_from_server', {'server_time': time.time()})
            socketio.sleep(5)
        except Exception as e:
            print(f"Error in server_ping: {e}")
            socketio.sleep(5)

@socketio.on('ping_from_client')
def handle_ping():
    socketio.emit('pong_from_server', {'server_time': time.time()})

def process_frame():
    while True:
        try:
            stream_manager.init_stream()
            results = detector.process_video_stream(
                        video_source=VIDEO_FILE,
                        max_frames=MAX_FRAMES
            )
            if results:
                print(results)
                socketio.emit("frame_processed", results)
            
            socketio.sleep(1)
        except Exception as e:
            print(f"Error in process_frame: {e}")
            socketio.sleep(5)



if __name__ == "__main__":
    socketio.start_background_task(server_ping)
    socketio.run(app, port=7860, host="0.0.0.0")
