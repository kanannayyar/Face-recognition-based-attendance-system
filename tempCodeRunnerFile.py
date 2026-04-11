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