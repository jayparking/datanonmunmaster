import os
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector


DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "dbname": "mydatabase",
    "user": "postgres",
    "password": "postgres",
}

K = 10

def get_class_from_path(p: str) -> str:
    # .../web_image/<class_name>/<file>.jpg 형태라고 가정
    parts = p.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"

conn = psycopg2.connect(**DB_CONFIG)
register_vector(conn)     # 🔥 반드시
cur = conn.cursor()

# id, class, text_emb, image_emb 로드
cur.execute("SELECT id, image_path, embedding, image_embedding FROM military_images;")
rows = cur.fetchall()
cur.close()
conn.close()

ids = []
classes = []
text_embs = []
img_embs = []

for _id, path, t, im in rows:
    if t is None or im is None:
        continue
    ids.append(_id)
    classes.append(get_class_from_path(path))
    text_embs.append(np.array(t, dtype=np.float32))
    img_embs.append(np.array(im, dtype=np.float32))

ids = np.array(ids)
classes = np.array(classes)
T = np.stack(text_embs)   # (N,512)
I = np.stack(img_embs)    # (N,512)

# cosine distance: (이미 L2 normalize 했으니) cosine sim = dot
def recall_at_k(emb: np.ndarray, k: int) -> float:
    N = emb.shape[0]
    hit = 0
    for i in range(N):
        sims = emb @ emb[i]          # (N,)
        sims[i] = -1e9               # 자기 자신 제외
        topk = np.argpartition(-sims, k)[:k]
        # top-k 중 같은 class가 하나라도 있으면 hit
        if np.any(classes[topk] == classes[i]):
            hit += 1
    return hit / N

r_text = recall_at_k(T, K)   # Ours: caption(text) embedding
r_img  = recall_at_k(I, K)   # Baseline: image-only embedding

print(f"Recall@{K} (Image-only CLIP) : {r_img:.4f}")
print(f"Recall@{K} (Caption-based)   : {r_text:.4f}")
