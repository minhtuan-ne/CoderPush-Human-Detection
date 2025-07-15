from deepface import DeepFace

class FaceAnalysis:
    def __init__(self):
        self.detector = 'opencv'
        self.enforce_detection = False
        self.align = True
        self.anti_spoofing = True

    def analyze_face(self, face):
        try:
            obj = DeepFace.analyze(
                img_path=face, 
                detector_backend=self.detector,
                enforce_detection=self.enforce_detection,
                align=self.align,
                anti_spoofing=self.anti_spoofing
            )
        except ValueError as e:
            if "Spoof detected" in str(e):
                print("Spoof detected, skipping this image.")
                return None 
            else:
                raise

        print(obj)
        print('Age: ', obj[0]['age'])
        print('Gender: ', obj[0]['dominant_gender'])
        print('Race: ', obj[0]['dominant_race'])
        print('Emotion: ', obj[0]['dominant_emotion'])
        return obj

if __name__ == "__main__":
    analyzer = FaceAnalysis()
    analyzer.analyze_face('detected_faces/face_3_20250714_103739.jpg')