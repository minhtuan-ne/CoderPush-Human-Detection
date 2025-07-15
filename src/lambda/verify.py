from deepface import DeepFace

def verify_face(face1, face2):
    detector = 'opencv'
    enforce_detection = False
    align = True
    anti_spoofing = True
    try:
        obj = DeepFace.verify(
            img1_path=face1,
            img2_path=face2,
            detector_backend=detector,
            enforce_detection=enforce_detection,
            align=align
        )
    except ValueError as e:
        if "Exception while processing" in str(e):
            print(f"Error processing image: {e}")
            return None
        else:
            raise
    
    if obj['verified']:
        print('Result: The images are of the same person.')
    else:
        print('Result: The images are of different people.')

    return obj 