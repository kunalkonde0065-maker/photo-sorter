import face_recognition
import os
import shutil
import numpy as np

INPUT_DIR = "input_photos"
OUTPUT_DIR = "output_photos"

TOLERANCE = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

known_faces = []
person_count = 0

def get_face_encodings(image_path):
    image = face_recognition.load_image_file(image_path)
    return face_recognition.face_encodings(image)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(INPUT_DIR, filename)
    face_encodings = get_face_encodings(image_path)

    if not face_encodings:
        unknown_dir = os.path.join(OUTPUT_DIR, "Unknown")
        os.makedirs(unknown_dir, exist_ok=True)
        shutil.move(image_path, os.path.join(unknown_dir, filename))
        continue

    matched_any = False

    for face_encoding in face_encodings:
        matched = False

        if known_faces:
            distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(distances)

            if distances[best_match_index] < TOLERANCE:
                person_dir = os.path.join(
                    OUTPUT_DIR, f"Person_{best_match_index + 1}"
                )
                os.makedirs(person_dir, exist_ok=True)
                shutil.copy(image_path, os.path.join(person_dir, filename))
                matched = True
                matched_any = True

        if not matched:
            person_count += 1
            known_faces.append(face_encoding)
            person_dir = os.path.join(OUTPUT_DIR, f"Person_{person_count}")
            os.makedirs(person_dir, exist_ok=True)
            shutil.copy(image_path, os.path.join(person_dir, filename))
            matched_any = True

    # Optional: move original only if at least one face matched
    if matched_any:
        os.remove(image_path)

print("✅ Photo sorting completed (multi-face supported)!")
