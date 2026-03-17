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

# Clear previous embeddings
faces_collection.delete_many({})
print("[INFO] Old embeddings removed from MongoDB.")

# ---------------------------
# Loop through dataset users
# ---------------------------
for user_name in os.listdir(dataset_path):

    user_folder = os.path.join(dataset_path, user_name)

    if not os.path.isdir(user_folder):
        continue

    print(f"\n[INFO] Processing user: {user_name}")

    user_embeddings = []

    # ---------------------------
    # Process all images of user
    # ---------------------------
    for file in os.listdir(user_folder):

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(user_folder, file)

        try:

            embedding_obj = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False
            )

            embedding = embedding_obj[0]["embedding"]

            # Normalize embedding (important for similarity comparison)
            embedding = np.array(embedding)
            embedding = embedding / np.linalg.norm(embedding)

            user_embeddings.append(embedding.tolist())

            print(f"✔ Processed {file}")

        except Exception as e:
            print(f"❌ Skipped {file} : {e}")

    # ---------------------------
    # Store user embeddings
    # ---------------------------
    if len(user_embeddings) > 0:

        faces_collection.insert_one({
            "user_name": user_name,
            "embeddings": user_embeddings,
            "total_images": len(user_embeddings),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"✅ Stored {len(user_embeddings)} embeddings for {user_name}")

    else:
        print(f"⚠ No valid images found for {user_name}")

# ---------------------------
# Training Complete
# ---------------------------
print("\n🎉 All embeddings stored successfully in MongoDB!")

client.close()