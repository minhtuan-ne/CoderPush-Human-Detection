import cv2
import face_recognition
from datetime import datetime
import os

class FaceDetector:
    def __init__(self, output_dir="detected_faces", tolerance=0.6):
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError("Tolerance must be between 0.0 and 1.0")
        self.face_counter = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.recent_encodings = []
        self.tolerance = tolerance 

    def detect_and_filter_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        new_faces = []
        for (location, encoding) in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.recent_encodings, encoding, tolerance=self.tolerance)
            if not any(matches):
                new_faces.append((location, encoding))
                self.recent_encodings.append(encoding)
                if len(self.recent_encodings) > 100:
                    self.recent_encodings = self.recent_encodings[-100:]
        return new_faces

    def process_video_stream(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame or reached end of video")
                break
            new_faces = self.detect_and_filter_faces(frame)
            for (top, right, bottom, left), encoding in new_faces:
                self.face_counter += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                margin = 30
                img_height, img_width = frame.shape[:2]
                crop_top = max(0, top - margin)
                crop_right = min(img_width, right + margin)
                crop_bottom = min(img_height, bottom + margin)
                crop_left = max(0, left - margin)
                face_img = frame[crop_top:crop_bottom, crop_left:crop_right]
                filename = f"face_{self.face_counter}_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                success = cv2.imwrite(filepath, face_img)
                if not success:
                    print(f"Failed to save face image to {filepath}")
                    continue
                print(f"Face #{self.face_counter} detected at {timestamp}, saved to {filepath}")
                # Here, send face_img, timestamp, and encoding to SQS for Lambda
                # (Can encode the image as base64 and the encoding as a list)
                # send_to_sqs(face_img, encoding, timestamp)
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceDetector()
    detector.process_video_stream()