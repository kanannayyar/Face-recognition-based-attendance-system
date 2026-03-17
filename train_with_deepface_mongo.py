from deepface import DeepFace
from pymongo import MongoClient
import numpy as np
import os
import time

# ---------------------------
# MongoDB Connection
# ---------------------------
# client = MongoClient("mongodb://localhost:27017/")
client = MongoClient("mongodb+srv://kananworks9_db_user:9rhFxzqwWhV9M3GO@cluster0.nd5mgtz.mongodb.net/?appName=Cluster0",tls=True,tlsAllowInvalidCertificates=True)
db = client["face_attendance"]
faces_collection = db["faces"]

# Dataset path
dataset_path = "face_dataset"


# ---------------------------
# Loop through dataset users
# ---------------------------
for user_name in os.listdir(dataset_path):

    user_folder = os.path.join(dataset_path, user_name)

    if not os.path.isdir(user_folder):
        continue

    # ✅ CHECK IF USER ALREADY EXISTS
    existing_user = faces_collection.find_one({"user_name": user_name})

    if existing_user:
        print(f"[SKIPPED] {user_name} already exists in DB")
        continue

    print(f"\n[INFO] Processing NEW user: {user_name}")

    user_embeddings = []

    # 🔹 LOOP THROUGH IMAGES
    for img_name in os.listdir(user_folder):

        img_path = os.path.join(user_folder, img_name)

        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False
            )

            embedding = result[0]["embedding"]
            user_embeddings.append(embedding)

        except Exception as e:
            print(f"[ERROR] {img_name} skipped: {e}")

    # ✅ INSERT ONLY IF EMBEDDINGS FOUND
    if len(user_embeddings) > 0:

        faces_collection.insert_one({
            "user_name": user_name,
            "embeddings": user_embeddings,
            "total_images": len(user_embeddings),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"[SAVED] {user_name} added with {len(user_embeddings)} embeddings")

    else:
        print(f"[WARNING] No valid images for {user_name}")
        
# ---------------------------
# Training Complete
# ---------------------------
print("\n🎉 All embeddings stored successfully in MongoDB!")

client.close()