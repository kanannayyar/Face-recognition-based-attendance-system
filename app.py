from flask import Flask, render_template, Response, request, redirect, session
import cv2
from deepface import DeepFace
from pymongo import MongoClient
import numpy as np
from datetime import datetime
import csv
import os
import time
import bcrypt
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
faculty_collection = db["faculty"]

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

@app.route("/set_filters", methods=["POST"])
def set_filters():
    data = request.json

    session["branch"] = data.get("branch")
    session["batch"] = data.get("batch")
    session["group"] = data.get("group")
    session["semester"] = data.get("semester")

    return {"status": "ok"}

# ---------------- LOAD EMBEDDINGS ----------------
def load_embeddings(filters=None):

    query = filters if filters else {}

    docs = list(faces_collection.find(query, {"_id":0}))

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

@app.route("/reload_embeddings", methods=["POST"])
def reload_embeddings():
    global names, embeddings

    filters = {}

    if session.get("branch"):
        filters["branch"] = session["branch"]
    if session.get("batch"):
        filters["batch"] = session["batch"]
    if session.get("group"):
        filters["group"] = session["group"]
    if session.get("semester"):
        filters["semester"] = session["semester"]

    names, embeddings = load_embeddings(filters)

    return {"status": "reloaded"}

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

    if not embeddings:
        return None, 0

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

            # run recognition every 90 frames
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

    return render_template(
        "dashboard.html",
        faculty_name=session.get("name"),
        department=session.get("department"),
        designation=session.get("designation")
    )


@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        # 🔍 find user by email
        user = faculty_collection.find_one({"email": email})

        if user:
            stored_password = user["password"]

            # ✅ compare hashed password
            if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):

                session["user"] = email
                session["name"] = user.get("name", "")
                session["department"] = user.get("department", "")
                session["designation"] = user.get("designation", "")

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

@app.route("/get_recognition")
def get_recognition():
    global last_name, last_similarity, last_seen_time

    if last_name and time.time() - last_seen_time < 10:

        doc = faces_collection.find_one({"user_name": last_name})

        return {
            "status": "success",
            "name": doc.get("name", last_name),
            "rollno": doc.get("rollno", ""),
            "branch": doc.get("branch", ""),
            "similarity": round(last_similarity, 2)
        }

    return {"status": "waiting"}

@app.route("/get_filters")
def get_filters():

    branches = faces_collection.distinct("branch")
    batches = faces_collection.distinct("batch")
    groups = faces_collection.distinct("group")
    semesters = faces_collection.distinct("semester")

    return {
        "branches": branches,
        "batches": batches,
        "groups": groups,
        "semesters": semesters
    }

@app.route("/get_total_students")
def get_total_students():

    query = {}

    # apply same filters
    if session.get("branch"):
        query["branch"] = session["branch"]
    if session.get("batch"):
        query["batch"] = session["batch"]
    if session.get("group"):
        query["group"] = session["group"]
    if session.get("semester"):
        query["semester"] = session["semester"]

    count = faces_collection.count_documents(query)

    return {"total": count}

@app.route("/get_students")
def get_students():

    query = {}

    # apply filters from session
    if session.get("branch"):
        query["branch"] = session["branch"]
    if session.get("batch"):
        query["batch"] = session["batch"]
    if session.get("group"):
        query["group"] = session["group"]
    if session.get("semester"):
        query["semester"] = session["semester"]

    docs = list(faces_collection.find(query, {"_id": 0}))

    students = []

    for d in docs:
        students.append({
            "roll": d.get("rollno", ""),
            "name": d.get("name", ""),
            "email": d.get("email", ""),
            "status": "not-marked"
        })

    return {"students": students}


@app.route("/students")
def students_page():
    if "user" not in session:
        return redirect("/login")

    return render_template(
        "student.html",
        faculty_name=session.get("name")
    )
# ---------------- RUN SERVER ----------------

if __name__=="__main__":

    app.run(host="0.0.0.0",port=5000,debug=True)