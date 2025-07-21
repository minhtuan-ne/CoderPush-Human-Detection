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
CORS(app)
detector = FaceDetector()

stream_url = "https://www.youtube.com/watch?v=cH7VBI4QQzA"
output_file = "live.ts"
subprocess.Popen(["streamlink", "--force", stream_url, "best", "-o", output_file])

@app.route('/api/process_stream', methods=['GET'])
def process_stream():
    results = detector.process_video_stream(video_source=output_file, show_window=False, max_frames=10)
    return jsonify(results)

# @app.route('/api/detect', methods=['POST'])
# def detect():
#     file_bytes = request.get_data()
#     npimg = np.frombuffer(file_bytes, np.uint8)
#     frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#     if frame is None:
#         return jsonify({'error': 'Invalid image'}), 400
#     results = detector.process_frame(frame)
#     for result in results:
#         analysis = analyze_face(result["img_URL"])
#         if "error" in analysis:
#             result["emotion_error"] = 'Cannot detect'
#         else:
#             result["emotion"] = analysis["emotion"]
#             result["emotion_confidence"] = analysis["emotion_confidence"]
#     return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=7860)