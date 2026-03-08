
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis




app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))



def average_embeddings(embeddings):
    avg_emb = embeddings.mean(axis=0) # Compute mean embedding
    avg_emb = avg_emb / np.linalg.norm(avg_emb) # Normalize 
    return avg_emb


def Embedding(folder):
    emb = []
    
    for filename in os.listdir(folder):
        if not filename.endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder, filename)
 
        img = cv2.imread(img_path)  # read image in BGR format

        if img is None:
            continue
 
        faces = app.get(img)
        if len(faces) == 0:
            continue
 
        face = faces[0] # Assuming one face per image
        embedding = face.normed_embedding  # L2-normalized embedding
        emb.append(embedding)

        if len(emb) == 0:  # Limit to 5 images per person for registration
            raise ValueError(f"No valid faces found in {folder}.")

    np_emb = np.array(emb)  # Convert list to NumPy array
    
    # print(f"Embeddings shape: {np_emb.shape}") # kept for debugging 

    avg_emb = average_embeddings(np_emb)  # Compute average embedding and normalize

     # print("Avg emb norm:", np.linalg.norm(avg_emb))  # kept for debugging
 
    return avg_emb 

  # return avg_emb,face

# reg_folder= r"E:\Projects\FaceRecognition\Data\Registration"

# embedding_store , extra = Embedding(reg_folder)

# np.save("PersonA_embedding.npy", embedding_store)