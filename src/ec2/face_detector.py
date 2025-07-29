import cv2
from datetime import datetime, timezone
import os
import numpy as np
from insightface.app import FaceAnalysis
from zoneinfo import ZoneInfo
from numpy.linalg import norm
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

class FaceDetector:
    def __init__(self, output_dir="detected_faces", tolerance=0.6, s3_bucket=None, s3_prefix="faces/"):
        self.face_counter = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.tolerance = tolerance
        self.known_embeddings = []
        self.models = ['buffalo_l', 'buffalo_m', 'buffalo_s', 'buffalo_sc', 'antelopev2']
        self.current_model = self.models[0]
        
        # S3 configuration
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = None
        
        # Initialize S3 client if bucket is provided
        if self.s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
                # Test S3 connection
                self.s3_client.head_bucket(Bucket=self.s3_bucket)
                print(f"S3 bucket '{self.s3_bucket}' is accessible")
            except NoCredentialsError:
                print("AWS credentials not found. S3 upload will be disabled.")
                self.s3_client = None
            except ClientError as e:
                print(f"S3 bucket access error: {e}. S3 upload will be disabled.")
                self.s3_client = None
        
        try:
            self.face_app = FaceAnalysis(name=self.current_model, providers=['CPUExecutionProvider'], root="/tmp/.insightface")
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize InsightFace: {str(e)}")

        self.frame_skip_counter = 0

    def get_local_timestamp(self, tz_str='Asia/Ho_Chi_Minh'):
        utc_now = datetime.now(timezone.utc)
        local_time = utc_now.astimezone(ZoneInfo(tz_str))
        return local_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    def upload_to_s3(self, local_filepath, s3_key):
        if not self.s3_client or not self.s3_bucket:
            return None
        
        try:
            self.s3_client.upload_file(
                local_filepath, 
                self.s3_bucket, 
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/jpeg',
                    'Metadata': {
                        'uploaded_at': datetime.utcnow().isoformat(),
                        'source': 'face_detector'
                    }
                }
            )
            s3_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
            print(f"Uploaded to S3: {s3_url}")
            return s3_url
        except ClientError as e:
            print(f"Failed to upload {local_filepath} to S3: {e}")
            return None

    def is_duplicate(self, embedding):
        for known_emb in self.known_embeddings:
            sim = np.dot(embedding, known_emb) / (norm(embedding) * norm(known_emb))
            if sim > (1 - self.tolerance):
                print("Duplicate face")
                return True
        return False

    def process_frame(self, frame):
        self.frame_skip_counter += 1
        if self.frame_skip_counter % 20 != 0:
            return []

        # else:
        #     filename = f"frame_{self.frame_skip_counter}.jpg"
        #     filepath = os.path.join('saved_frames', filename)
        #     os.makedirs('saved_frames', exist_ok=True)
        #     success = cv2.imwrite(filepath, frame)

        print("Processing frame", self.frame_skip_counter)

        faces = self.face_app.get(frame)
        results = []
        for face in faces:
            embedding = face.embedding
            if embedding is None:
                continue
            if self.is_duplicate(embedding):
                continue
            self.known_embeddings.append(embedding)
            self.face_counter += 1
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            timestamp = self.get_local_timestamp()
            
            # Crop and save face locally
            local_filepath = self._crop_and_save_face(frame, y1, x2, y2, x1, self.face_counter, timestamp)
            if not local_filepath:
                continue
            
            # Upload to S3 if configured
            s3_url = None
            if self.s3_client and self.s3_bucket:
                safe_timestamp = timestamp.replace(":", "-")
                s3_key = f"{self.s3_prefix}face_{self.face_counter}_{safe_timestamp}.jpg"
                s3_url = self.upload_to_s3(local_filepath, s3_key)
            
            results.append({
                "face_id": self.face_counter,
                "img_URL": s3_url,
                "timestamp": timestamp
            })
        return results

    def process_video_stream(self, video_source=0, show_window=False, max_frames=10):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")
        frame_count = 0
        all_results = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break
                results = self.process_frame(frame)

                if results:
                    all_results.extend(results)

                frame_count += 1

                if show_window:
                    cv2.imshow('Face Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception as e:
            print(f"Error during video processing: {e}")
            raise
        finally:
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

        # Compress image before saving
        success, encoded_img = cv2.imencode('.jpg', face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if success:
            with open(filepath, 'wb') as f:
                f.write(encoded_img)
            return filepath
        return None

    def cleanup_local_files(self, keep_local=False):
        if not keep_local and self.s3_client:
            try:
                for filename in os.listdir(self.output_dir):
                    if filename.endswith('.jpg'):
                        os.remove(os.path.join(self.output_dir, filename))
                        print(f"Cleaned up local file: {filename}")
            except Exception as e:
                print(f"Error cleaning up local files: {e}")


# if __name__ == "__main__":
#     detector = FaceDetector()
#     # frame = cv2.imread('detected_faces/sample_frame.png')
#     # if frame is None:
#     #     print("Failed to load image!")
#     # else:
#         # detector.process_frame(frame)
#     video_source = 'live.ts'
#     detector.process_video_stream(video_source=video_source)