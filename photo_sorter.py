import face_recognition
import os
import shutil
import cv2

INPUT_DIR = r"C:\Users\kunal\OneDrive\Desktop\Photo sorter"
OUTPUT_DIR = "output_photos"

os.makedirs(OUTPUT_DIR, exist_ok=True)

known_faces = []
person_count = 0

def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(INPUT_DIR, filename)
    face_encoding = get_face_encoding(image_path)

    if face_encoding is None:
        unknown_dir = os.path.join(OUTPUT_DIR, "Unknown")
        os.makedirs(unknown_dir, exist_ok=True)
        shutil.move(image_path, os.path.join(unknown_dir, filename))
        continue

    matched = False

    for i, known_face in enumerate(known_faces):
        match = face_recognition.compare_faces([known_face], face_encoding)[0]
        if match:
            person_dir = os.path.join(OUTPUT_DIR, f"Person_{i+1}")
            shutil.move(image_path, os.path.join(person_dir, filename))
            matched = True
            break

    if not matched:
        person_count += 1
        known_faces.append(face_encoding)
        person_dir = os.path.join(OUTPUT_DIR, f"Person_{person_count}")
        os.makedirs(person_dir, exist_ok=True)
        shutil.move(image_path, os.path.join(person_dir, filename))

print("✅ Photo sorting completed!")

