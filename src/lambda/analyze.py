from deepface import DeepFace

def analyze_face(face):
    detector = 'opencv'
    enforce_detection = False
    align = True
    
    if not face:
        raise ValueError("Face image path cannot be empty")
    try:
        obj = DeepFace.analyze(
            img_path=face,
            detector_backend=detector,
            enforce_detection=enforce_detection,
            align=align
        )
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    
    if not obj:
        return None

    print('Age:', obj[0]['age'])
    print('Gender:', obj[0]['dominant_gender'])
    print('Race:', obj[0]['dominant_race'])
    print('Emotion:', obj[0]['dominant_emotion'])

    return obj 