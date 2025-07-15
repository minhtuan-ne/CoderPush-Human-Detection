from deepface import DeepFace

class FaceAnalysis:
    def __init__(self):
        pass

    def analyze_face(self, face):
        align = True
        detector = 'opencv'
        obj = DeepFace.analyze(img_path=face, 
                               detector_backend=detector,
                               enforce_detection=False,
                               align=align)
        print(obj)
        print('Age: ', obj[0]['age'])
        print('Gender: ', obj[0]['dominant_gender'])
        print('Race: ', obj[0]['dominant_race'])
        print('Emotion: ', obj[0]['dominant_emotion'])
        return obj
    
if __name__ == "__main__":
    analyzer = FaceAnalysis()
    analyzer.analyze_face('detected_faces/face_3_20250714_103705.jpg')