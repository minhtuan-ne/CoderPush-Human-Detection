from deepface import DeepFace

def verify_face(face1: str, face2: str) -> dict:
    """
    Verify if two face images belong to the same person and return a JSON-compatible dict with the result.
    Args:
        face1 (str): Path to the first face image file.
        face2 (str): Path to the second face image file.
    Returns:
        dict: Verification result including samePerson, verification_confidence, and error if any.
    """
    detector = 'opencv'
    enforce_detection = False
    align = True

    if not face1 or not face2:
        return {"error": "Both face image paths must be provided"}
    try:
        obj = DeepFace.verify(
            img1_path=face1,
            img2_path=face2,
            detector_backend=detector,
            enforce_detection=enforce_detection,
            align=align
        )
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": str(e)}

    if not obj:
        return {"error": "No verification result returned"}

    verification_result = {
        'samePerson': bool(obj['verified']),
        'verification_confidence': max(0.0, float(1 - obj.get('distance', 1))),
    }

    return verification_result 