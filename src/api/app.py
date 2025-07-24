from flask import Flask, request, jsonify
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from ec2.face_detector import FaceDetector
from lambda_aws.analyze import analyze_face
from flask_cors import CORS
import subprocess


app = Flask(__name__)
CORS(app, origins=["*"])
detector = FaceDetector()

stream_url_1 = "https://www.youtube.com/watch?v=cH7VBI4QQzA"
stream_url_2 = "https://www.youtube.com/watch?v=VR-x3HdhKLQ"
output_file = "live.ts"
subprocess.Popen(["streamlink", "--force", stream_url_2, "best", "-o", output_file])

@app.route('/')
def home():
    """A simple health check endpoint."""
    return jsonify({"status": "ok", "message": "API is running"})

@app.route('/api/process_stream', methods=['GET'])
def process_stream():
    results = detector.process_video_stream(video_source=output_file, show_window=False, max_frames=10)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=7860)