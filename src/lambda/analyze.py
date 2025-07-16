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
    
    dominant_emotion = obj[0]['dominant_emotion']
    emotion_confidence = obj[0]['emotion'][dominant_emotion]

    analysis_result = {
        'emotion': dominant_emotion,
        'emotion_confidence': emotion_confidence,
    }

    return analysis_result 