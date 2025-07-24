from flask import Flask, request, jsonify
import numpy as np
import cv2
import sys
import os
import time
import atexit
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from ec2.face_detector import FaceDetector
from flask_cors import CORS
from stream_manager import StreamManager

app = Flask(__name__)
CORS(app, origins=["*"])

# Initialize components
detector = FaceDetector()

# Stream URLs
STREAM_URL_1 = "https://www.youtube.com/watch?v=cH7VBI4QQzA"
STREAM_URL_2 = "https://www.youtube.com/watch?v=VR-x3HdhKLQ"
OUTPUT_FILE = "live.ts"

# Initialize stream manager
stream_manager = StreamManager(
    stream_url=STREAM_URL_2,
    output_file=OUTPUT_FILE,
    max_wait_time=30
)

@app.route('/')
def home():
    """Health check endpoint with stream status."""
    status = stream_manager.get_status()
    return jsonify({
        "status": "ok",
        "message": "API is running",
        "stream_status": status
    })

@app.route('/api/stream/status', methods=['GET'])
def stream_status():
    """Get detailed stream status."""
    return jsonify(stream_manager.get_status())

@app.route('/api/stream/init', methods=['POST'])
def initialize_stream():
    """Initialize or restart the stream."""
    success = stream_manager.init_stream()
    return jsonify({
        "success": success,
        "status": stream_manager.get_status()
    })

@app.route('/api/stream/restart', methods=['POST'])
def restart_stream():
    """Restart the stream."""
    success = stream_manager.restart()
    return jsonify({
        "success": success,
        "status": stream_manager.get_status()
    })

@app.route('/api/process_stream', methods=['GET'])
def process_stream():
    """Process the video stream for face detection."""
    # Check if stream is ready
    if not stream_manager.is_ready:
        return jsonify({
            "error": "Stream not ready",
            "status": stream_manager.get_status(),
            "suggestion": "Call /api/stream/init first"
        }), 400
    
    # Check stream health before processing
    if not stream_manager.is_healthy():
        # Try to restart stream
        app.logger.info("Stream unhealthy, attempting restart...")
        if stream_manager.restart():
            time.sleep(2)  # Give it a moment to stabilize
            app.logger.info("Stream restarted successfully")
        else:
            return jsonify({
                "error": "Stream is unhealthy and failed to restart",
                "status": stream_manager.get_status()
            }), 500
    
    try:
        results = detector.process_video_stream(
            video_source=OUTPUT_FILE, 
            show_window=False, 
            max_frames=10
        )
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error processing stream: {str(e)}")
        return jsonify({
            "error": f"Error processing stream: {str(e)}",
            "stream_status": stream_manager.get_status()
        }), 500


def cleanup_on_exit():
    """Clean up resources on application exit."""
    stream_manager.cleanup()

if __name__ == '__main__':
    atexit.register(cleanup_on_exit)
    
    stream_manager.init_async()
    
    app.run(debug=True, host="0.0.0.0", port=7860)