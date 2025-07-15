from analyze import analyze_face
from verify import verify_face

if __name__ == "__main__":
    # Example usage: analyze a face
    print('Example: FACE ANALYSIS:')
    analyze_face('detected_faces/face_1_20250714_103651.jpg')
    # Example usage: verify two faces
    print('Example: FACE VERIFICATION:')
    verify_face('detected_faces/face_1_20250714_103651.jpg', 'detected_faces/face_1_20250714_103731.jpg')