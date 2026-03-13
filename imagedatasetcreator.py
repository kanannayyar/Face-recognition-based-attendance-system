import cv2
import time
import os

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Create a folder to store face datasets
dataset_path = "face_dataset"
os.makedirs(dataset_path, exist_ok=True)

# Ask for User ID or Name
user_name = input("Enter the user name or ID: ").strip()
user_folder = os.path.join(dataset_path, user_name)
os.makedirs(user_folder, exist_ok=True)

cap = cv2.VideoCapture(0)

blink_detected = False
eyes_last_seen = time.time()
BLINK_TIMEOUT = 5
capture_count = 0
MAX_CAPTURES = 30  # number of face images to save

print("[INFO] Starting camera... Blink once to confirm liveness.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    status_text = "No face detected"
    text_color = (255, 255, 255)

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        (x, y, w, h) = faces[0]

        # Detect eyes in the face region
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        current_time = time.time()

        # Blink detection
        if len(eyes) >= 1:
            eyes_last_seen = current_time
            if blink_detected and capture_count < MAX_CAPTURES:
                # Capture and save face
                face_crop = gray[y:y+h, x:x+w]
                file_name = os.path.join(user_folder, f"{user_name}_{capture_count+1}.jpg")
                cv2.imwrite(file_name, face_crop)
                capture_count += 1
                status_text = f"Capturing... {capture_count}/{MAX_CAPTURES}"
                text_color = (0, 255, 0)
            elif capture_count >= MAX_CAPTURES:
                status_text = "✅ Dataset captured successfully!"
                text_color = (0, 255, 0)
        else:
            # Eyes disappeared → blink detected
            if current_time - eyes_last_seen < 1.0 and not blink_detected:
                blink_detected = True
                status_text = "✅ Blink detected! Capturing faces..."
                text_color = (0, 255, 0)
            elif current_time - eyes_last_seen > BLINK_TIMEOUT and not blink_detected:
                status_text = "🚫 No blink detected (possible photo)"
                text_color = (0, 0, 255)

        # Draw white corners
        line_len = 30
        cv2.line(frame, (x, y), (x + line_len, y), (255, 255, 255), 2)
        cv2.line(frame, (x, y), (x, y + line_len), (255, 255, 255), 2)
        cv2.line(frame, (x + w, y), (x + w - line_len, y), (255, 255, 255), 2)
        cv2.line(frame, (x + w, y), (x + w, y + line_len), (255, 255, 255), 2)
        cv2.line(frame, (x, y + h), (x + line_len, y + h), (255, 255, 255), 2)
        cv2.line(frame, (x, y + h), (x, y + h - line_len), (255, 255, 255), 2)
        cv2.line(frame, (x + w, y + h), (x + w - line_len, y + h), (255, 255, 255), 2)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - line_len), (255, 255, 255), 2)

        # Draw status text
        cv2.putText(frame, status_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    cv2.imshow("Liveness & Face Capture", frame)

    # Exit if 'q' pressed or all images captured
    if cv2.waitKey(1) & 0xFF == ord('q') or capture_count >= MAX_CAPTURES:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] {capture_count} face images saved in {user_folder}")
