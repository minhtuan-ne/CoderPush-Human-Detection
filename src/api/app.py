from flask import Flask, jsonify
from stream_manager import StreamManager
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from ec2.face_detector import FaceDetector
import os
import logging

app = Flask(__name__)

# Configuration
STREAM_URL_1 = "https://www.youtube.com/watch?v=cH7VBI4QQzA"
STREAM_URL_2 = "https://www.youtube.com/watch?v=VR-x3HdhKLQ"
VIDEO_FILE = "live.ts"
MAX_FRAMES = 200

# Initialize components
stream_manager = StreamManager(STREAM_URL_2, output_file=VIDEO_FILE)
detector = FaceDetector()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/api/process_stream", methods=["GET"])
def process_stream():
    stream_manager.init_stream()
    results = detector.process_video_stream(video_source=VIDEO_FILE, max_frames=MAX_FRAMES)
    return jsonify(results)


@app.route("/api/stream_status", methods=["GET"])
def stream_status():
    return jsonify(stream_manager.get_status())


@app.route("/api/stream_restart", methods=["POST"])
def stream_restart():
    success = stream_manager.restart()
    return jsonify({
        "success": success,
        "error": stream_manager.error if not success else None
    })


@app.route("/api/stream_cleanup", methods=["POST"])
def stream_cleanup():
    stream_manager.cleanup()
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True, port='7860', host='0.0.0.0')
