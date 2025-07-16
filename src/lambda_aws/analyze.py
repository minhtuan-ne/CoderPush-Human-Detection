from deepface import DeepFace

def analyze_face(face: str) -> dict:
    """
    Analyze a face image and return a JSON-compatible dict with analysis results.
    Args:
        face (str): Path to the face image file.
    Returns:
        dict: Analysis result including emotion and emotion_confidence, or error message.
    """
    detector = 'opencv'
    enforce_detection = False
    align = True

    if not face:
        return {"error": "Face image path cannot be empty"}
    try:
        obj = DeepFace.analyze(
            img_path=face,
            detector_backend=detector,
            enforce_detection=enforce_detection,
            align=align
        )
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": str(e)}

    if not obj:
        return {"error": "No analysis result returned"}

    dominant_emotion = obj[0]['dominant_emotion']
    emotion_confidence = obj[0]['emotion'][dominant_emotion]

    analysis_result = {
        'emotion': dominant_emotion,
        'emotion_confidence': emotion_confidence,
    }

    return analysis_result 