import numpy as np
import psycopg2
import torch
from transformers import CLIPProcessor, CLIPModel
from pgvector.psycopg2 import register_vector

# =========================
# 설정
# =========================
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "dbname": "mydatabase",
    "user": "postgres",
    "password": "postgres",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "openai/clip-vit-base-patch32"

KS = [1, 5, 10]

# 각 이미지 클래스에 대응되는 "단순 키워드"
# 👉 필요하면 더 추가해도 됨
CLASS_TO_KEYWORD = {
    "fixed-wing combat aircraft": "fighter aircraft",
    "tank": "tank",
    "missile": "missile",
    "naval ship": "warship",
}

# =========================
# 1. CLIP 로드
# =========================
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
clip_model.eval()

def embed_text(text: str) -> np.ndarray:
    inputs = clip_processor(text=[text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

# =========================
# 2. DB에서 image embedding 로드
# =========================
conn = psycopg2.connect(**DB_CONFIG)
register_vector(conn)
cur = conn.cursor()

cur.execute("""
    SELECT id, image_path, image_embedding
    FROM military_images
    WHERE image_embedding IS NOT NULL
    ORDER BY id;
""")

rows = cur.fetchall()
cur.close()
conn.close()

ids = []
classes = []
I = []   # image embeddings

import os

def get_class_from_path(p: str) -> str:
    name = os.path.basename(p)          # 파일명만 추출
    name = os.path.splitext(name)[0]    # .jpg 제거
    return name.split("_")[0]            # 앞부분 = 클래스


for _id, path, im in rows:
    cls = get_class_from_path(path)
    if cls not in CLASS_TO_KEYWORD:
        continue
    ids.append(_id)
    classes.append(cls)
    I.append(np.array(im, dtype=np.float32))
    print("DEBUG path:", path)
    print("DEBUG class:", cls)


ids = np.array(ids)
classes = np.array(classes)
I = np.stack(I)     # (N, 512)

N = I.shape[0]
print(f"Loaded {N} images for keyword baseline")

# =========================
# 3. Keyword baseline Recall@K
# =========================
def recall_at_k_keyword(I, classes, ks):
    recalls = {k: 0 for k in ks}

    for i in range(N):
        keyword = CLASS_TO_KEYWORD[classes[i]]
        q = embed_text(keyword)

        sims = I @ q          # cosine similarity
        rank = np.argsort(-sims)

        for k in ks:
            if i in rank[:k]:
                recalls[k] += 1

    for k in recalls:
        recalls[k] /= N

    return recalls

results = recall_at_k_keyword(I, classes, KS)

# =========================
# 4. 결과 출력
# =========================
print("\n=== Keyword Baseline: Text → Image (GT pair) ===")
for k in KS:
    print(f"Recall@{k}: {results[k]:.4f}")
