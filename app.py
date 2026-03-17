from flask import Flask, render_template, Response, request, redirect, session
import cv2
from deepface import DeepFace
from pymongo import MongoClient
import numpy as np
from datetime import datetime
import csv
import os
import time
print("[INFO] Loading FaceNet model...")
DeepFace.build_model("Facenet")
print("[INFO] Model loaded successfully!")

app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- CAMERA ----------------
camera = None

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- DATABASE ----------------
# client = MongoClient("mongodb://localhost:27017/")
client = MongoClient("mongodb+srv://kananworks9_db_user:9rhFxzqwWhV9M3GO@cluster0.nd5mgtz.mongodb.net/?appName=Cluster0",tls=True,tlsAllowInvalidCertificates=True)
db = client["face_attendance"]
faces_collection = db["faces"]

SIMILARITY_THRESHOLD = 0.45

marked_today = set()

# -------- recognition cache --------
last_name = None
last_similarity = 0
last_seen_time = 0

# ---------------- ATTENDANCE FILE ----------------
attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_name","date","time","similarity"])

# ---------------- LOAD EMBEDDINGS ----------------
def load_embeddings():

    docs = list(faces_collection.find({}, {"_id":0}))

    names = []
    embeddings = []

    for doc in docs:

        user = doc["user_name"]

        for emb in doc["embeddings"]:
            names.append(user)
            embeddings.append(np.array(emb))

    print(f"[INFO] Loaded {len(embeddings)} embeddings")

    return names, embeddings

names, embeddings = load_embeddings()

# ---------------- SIMILARITY ----------------
def cosine_similarity(a,b):

    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# ---------------- FACE RECOGNITION ----------------
def recognize_face(face_img):

    try:

        result = DeepFace.represent(
            img_path=face_img,
            model_name="Facenet",
            enforce_detection=False
        )

        emb = np.array(result[0]["embedding"])
        emb = emb / np.linalg.norm(emb)

    except:
        return None,0

    best_sim = -1
    best_name = None

    for name,ref in zip(names,embeddings):

        sim = cosine_similarity(emb,ref)

        if sim > best_sim:
            best_sim = sim
            best_name = name

    if best_sim > SIMILARITY_THRESHOLD:
        return best_name,best_sim

    return None,best_sim


# ---------------- VIDEO STREAM ----------------
def gen_frames():

    global camera
    global last_name,last_similarity,last_seen_time

    # reopen camera if stopped
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

    frame_count = 0

    while True:

        if camera is None:
            break

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame,1)

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.2,6)

        if len(faces) > 0:

            x,y,w,h = max(faces,key=lambda b:b[2]*b[3])

            face_img = frame[y:y+h,x:x+w]

            frame_count += 1

            # run recognition every 30 frames
            if frame_count % 30 == 0:

                name,sim = recognize_face(face_img)

                if name:

                    last_name = name
                    last_similarity = sim
                    last_seen_time = time.time()

                    today = datetime.now().strftime("%Y-%m-%d")

                    if (name,today) not in marked_today:

                        time_now = datetime.now().strftime("%H:%M:%S")

                        with open(attendance_file,"a",newline="") as f:

                            writer = csv.writer(f)

                            writer.writerow([
                                name,
                                today,
                                time_now,
                                round(sim,3)
                            ])

                        marked_today.add((name,today))

                        print(f"[ATTENDANCE] {name} marked")

            # display cached result
            if last_name and time.time()-last_seen_time < 3:

                label=f"{last_name} ({last_similarity:.2f})"
                color=(0,255,0)

            else:

                label="Unknown"
                color=(0,0,255)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            cv2.putText(
                frame,
                label,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

# ---------------- ROUTES ----------------

@app.route("/")
def home():

    if "user" not in session:
        return redirect("/login")

    return render_template("dashboard.html")


@app.route("/login",methods=["GET","POST"])
def login():

    if request.method=="POST":

        email=request.form["email"]
        password=request.form["password"]

        if email=="admin@gmail.com" and password=="12345":

            session["user"]=email
            return redirect("/")

        if email=="kanannayyar9@gmail.com" and password=="09211":

            session["user"]=email
            return redirect("/")

        return "Invalid Credentials"

    return render_template("login.html")


@app.route("/logout")
def logout():

    session.pop("user",None)
    return redirect("/login")


@app.route('/video')
def video():

    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stop')
def stop_camera():

    global camera

    if camera is not None:

        camera.release()
        camera = None

    return "Camera stopped"


# ---------------- RUN SERVER ----------------

if __name__=="__main__":

    app.run(host="0.0.0.0",port=5000,debug=True)