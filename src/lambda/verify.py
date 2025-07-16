from deepface import DeepFace

def verify_face(face1, face2):
    detector = 'opencv'
    enforce_detection = False
    align = True
    
    if not face1 or not face2:
        raise ValueError("Both face image paths must be provided")
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
        return None
    
    if not obj:
        return None
    
    verification_result = {
        'verified': obj['verified'],
        'confidence': obj.get('distance', 0),
        'message': 'Same person' if obj['verified'] else 'Different people',
        'raw_data': obj
    }
    
    print(verification_result['message'])

    return verification_result 