import cv2
from datetime import datetime, timezone
import os
import numpy as np
from insightface.app import FaceAnalysis
from zoneinfo import ZoneInfo
from numpy.linalg import norm
# import time

class FaceDetector:
    def __init__(self, output_dir="detected_faces", tolerance=0.6):
        self.face_counter = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.tolerance = tolerance
        self.known_embeddings = []
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'], root="/tmp/.insightface")
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def get_local_timestamp(self, tz_str='Asia/Ho_Chi_Minh'):
        utc_now = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        local_time = utc_now.astimezone(ZoneInfo(tz_str))
        return local_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    # def is_duplicate(self, embedding):
    #     for known_emb in self.known_embeddings:
    #         sim = np.dot(embedding, known_emb) / (norm(embedding) * norm(known_emb))
    #         if sim > (1 - self.tolerance):
    #             return True
    #     return False

    def process_frame(self, frame):
        faces = self.face_app.get(frame)
        results = []
        for face in faces:
            embedding = face.embedding
            if embedding is None:
                continue
            # if self.is_duplicate(embedding):
            #     continue
            self.known_embeddings.append(embedding)
            self.face_counter += 1
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            timestamp = self.get_local_timestamp()
            filepath = self._crop_and_save_face(frame, y1, x2, y2, x1, self.face_counter, timestamp)
            if not filepath:
                continue
            results.append({
                "face_id": self.face_counter,
                "img_URL": filepath,
                "timestamp": timestamp
            })
        print(results)
        return results

    # def process_video_stream(self, video_source=0, show_window=False, all_results=None, stop_event=None):
    #     """
    #     Continuously processes a video stream until a stop event is set.
    #     - video_source: Path to the video file or stream URL.
    #     - all_results: A thread-safe deque to append detection results to.
    #     - stop_event: A threading.Event to signal when to stop processing.
    #     """
    #     while stop_event is None or not stop_event.is_set():
    #         cap = cv2.VideoCapture(video_source)
    #         if not cap.isOpened():
    #             print(f"Failed to open video source: {video_source}, retrying in 5 seconds...")
    #             time.sleep(5)
    #             continue
    #
    #         print("Video source opened successfully. Starting frame processing.")
    #         while not stop_event.is_set():
    #             ret, frame = cap.read()
    #             if not ret:
    #                 print("End of stream or buffer. Re-opening video source...")
    #                 break  # Break inner loop to reopen the capture
    #
    #             results = self.process_frame(frame)
    #             if all_results is not None and results:
    #                 all_results.extend(results)
    #
    #             if show_window:
    #                 cv2.imshow('Face Detection', frame)
    #                 if cv2.waitKey(1) & 0xFF == ord('q'):
    #                     stop_event.set()
    #                     break
    #
    #         cap.release()
    #         if show_window:
    #             cv2.destroyAllWindows()
    #
    #     print("Video processing stopped.")

    def process_video_stream(self, video_source=0, show_window=False, max_frames=50):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")
        frame_count = 0
        all_results = []
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            results = self.process_frame(frame)
            all_results.extend(results)
            frame_count += 1
            if show_window:
                cv2.imshow('Face Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        return all_results

    def _crop_and_save_face(self, frame, top, right, bottom, left, face_id, timestamp):
        margin = 30
        img_height, img_width = frame.shape[:2]
        crop_top = max(0, top - margin)
        crop_right = min(img_width, right + margin)
        crop_bottom = min(img_height, bottom + margin)
        crop_left = max(0, left - margin)
        face_img = frame[crop_top:crop_bottom, crop_left:crop_right]
        safe_timestamp = timestamp.replace(":", "-")
        filename = f"face_{face_id}_{safe_timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        success = cv2.imwrite(filepath, face_img)
        return filepath if success else None

# if __name__ == "__main__":
#     detector = FaceDetector()
#     # frame = cv2.imread('detected_faces/sample_frame.png')
#     # if frame is None:
#     #     print("Failed to load image!")
#     # else:
#         # detector.process_frame(frame)
#     video_source = 'live.ts'
#     detector.process_video_stream(video_source=video_source)