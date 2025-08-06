from flask_socketio import SocketIO # type: ignore
from flask import Flask, request
import sys
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env/.env")

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from ec2.face_detector import FaceDetector

app = Flask(__name__)
socketio = SocketIO(app, 
                    cors_allowed_origins="*",
                    ping_timeout=250,
                    ping_interval=25)

# Configuration
STREAM_URL = os.getenv('STREAM_URL', "http://185.194.123.84:8001/mjpg/video.mjpg")
MAX_FRAMES = 100

# Initialize components with S3 configuration
detector = FaceDetector(
    s3_bucket=os.getenv('S3_BUCKET_NAME'),
    s3_prefix=os.getenv('S3_PREFIX', 'faces/')
)

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
            results = detector.process_video_stream(
                        video_source=STREAM_URL,
                        max_frames=MAX_FRAMES
            )
            if results:
                print(f"Detected {len(results)} faces")
                # Log S3 uploads
                for result in results:
                    if result.get('img_URL'):
                        print(f"Face {result['face_id']} uploaded to: {result['img_URL']}")
                
                socketio.emit("frame_processed", results)
            
            socketio.sleep(1)
        except Exception as e:
            print(f"Error in process_frame: {e}")
            socketio.sleep(5)

if __name__ == "__main__":
    s3_bucket = os.getenv('S3_BUCKET_NAME')
    if s3_bucket:
        print(f"S3 upload enabled - Bucket: {s3_bucket}")
    else:
        print("S3 upload disabled - no bucket configured")
    
    print("Staring serving ping")
    socketio.start_background_task(server_ping)
    print("Server is running")
    socketio.run(app, port=7860, host="0.0.0.0")