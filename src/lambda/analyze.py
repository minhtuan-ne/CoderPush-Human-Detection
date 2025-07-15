from deepface import DeepFace

def analyze_face(face):
    detector = 'opencv'
    enforce_detection = False
    align = True
    try:
        obj = DeepFace.analyze(
            img_path=face,
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

    print('Age: ', obj[0]['age'])
    print('Gender: ', obj[0]['dominant_gender'])
    print('Race: ', obj[0]['dominant_race'])
    print('Emotion: ', obj[0]['dominant_emotion'])

    return obj 