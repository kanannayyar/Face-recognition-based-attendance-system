from flask import Flask, render_template, Response, request, redirect, session
import cv2

app = Flask(__name__)
app.secret_key = "secret123"

camera = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def gen_frames():
    global camera

    camera = cv2.VideoCapture(0)

    while True:
        if camera is None:
            break

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame,1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame +
               b'\r\n')

@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("dashboard.html")

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        if email == "admin@gmail.com" and password == "12345":
            session["user"] = email
            return redirect("/")

        if email == "kanannayyar9@gmail.com" and password == "09211":
            session["user"] = email
            return redirect("/")

        return "Invalid Credentials"

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

@app.route('/video')
def video():
    return Response(gen_frames(),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop_camera():
    global camera

    if camera is not None:
        camera.release()
        camera = None

    return "Camera stopped"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)