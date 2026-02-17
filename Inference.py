import os
import numpy as np
from insightface.app import FaceAnalysis

# -------- Load InsightFace ONCE --------
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))


# -------- Load embeddings from subfolders --------
def load_embeddings(root_dir):
    names = []
    embeddings = []

    for person in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for file in os.listdir(person_dir):
            if file.endswith(".npy"):
                path = os.path.join(person_dir, file)

                emb = np.load(path).astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-8)

                embeddings.append(emb)
                names.append(person)

    if not embeddings:
        return [], None

    return names, np.vstack(embeddings)


# -------- Matcher Class --------
class FaceMatcher:
    def __init__(self, emb_root):
        self.names, self.db_embs = load_embeddings(emb_root)

    def match(self, frame):
        faces = app.get(frame)
        if not faces or self.db_embs is None:
            return None, None, None

        face = faces[0] # only one face 
        

        test_emb = face.normed_embedding.astype(np.float32)

        # Vectorized cosine similarity
        sims = np.dot(self.db_embs,test_emb)
        best_idx = int(np.argmax(sims))

        similarity = float(sims[best_idx])
        name = self.names[best_idx]

        x1, y1, x2, y2 = face.bbox.astype(int)
        bbox = [x1, y1, x2, y2]

        return similarity, name, bbox
