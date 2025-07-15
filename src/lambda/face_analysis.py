from analyze import analyze_face
from verify import verify_face

if __name__ == "__main__":
    # Replace the image path with your own path

    # Example usage: analyze a face
    print('Example: FACE ANALYSIS:')
    analyze_face('detected_faces/face_1_20250715_155915.jpg')
    analyze_face('detected_faces/face_2_20250715_160753.jpg') 

    # Example usage: verify two faces
    print('Example: FACE VERIFICATION:')
    verify_face('detected_faces/face_1_20250715_155915.jpg', 'detected_faces/face_1_20250715_160750.jpg')
    verify_face('detected_faces/face_1_20250715_155915.jpg', 'detected_faces/face_2_20250715_160753.jpg')