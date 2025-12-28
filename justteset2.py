import os
import numpy as np
import psycopg2
import torch

from transformers import CLIPProcessor, CLIPModel
from pgvector.psycopg2 import register_vector

# ============================================================
# 0. 설정
# ============================================================

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

# ============================================================
# 1. 비교 실험용 Query 정의
# ============================================================

# (1) 단순 키워드 baseline
KEYWORD_QUERIES = {
    "fixed-wing combat aircraft": "fighter aircraft",
    "tank": "tank",
    "missile": "missile",
    "naval ship": "warship",
}

# (2) 사람이 작성한 문장
HUMAN_SENTENCE_QUERIES = {
    "fixed-wing combat aircraft": "a fighter jet flying in the sky",
    "tank": "a main battle tank on the ground",
    "missile": "a missile launcher vehicle",
    "naval ship": "a naval warship sailing on the sea",
}

# ============================================================
# 2. CLIP 로드
# ============================================================

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
clip_model.eval()

def embed_text(text: str) -> np.ndarray:
    """Text → CLIP embedding (L2 normalized)"""
    inputs = clip_processor(text=[text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

# ============================================================
# 3. DB에서 image embedding + caption 로드
# ============================================================

def get_class_from_path(p: str) -> str:
    """파일명 앞부분을 class로 사용"""
    name = os.path.basename(p)
    name = os.path.splitext(name)[0]
    return name.split("_")[0]

conn = psycopg2.connect(**DB_CONFIG)
register_vector(conn)
cur = conn.cursor()

cur.execute("""
    SELECT id, image_path, caption, image_embedding
    FROM military_images
    WHERE image_embedding IS NOT NULL
    ORDER BY id;
""")

rows = cur.fetchall()
cur.close()
conn.close()

ids, classes, captions, I = [], [], [], []

for _id, path, caption, im in rows:
    cls = get_class_from_path(path)

    if cls not in KEYWORD_QUERIES:
        continue

    ids.append(_id)
    classes.append(cls)
    captions.append(caption)
    I.append(np.array(im, dtype=np.float32))

ids = np.array(ids)
classes = np.array(classes)
captions = np.array(captions)
I = np.stack(I)

N = I.shape[0]
print(f"Loaded {N} images for comparison experiment")

# ============================================================
# 4. Recall@K 계산 함수 (공통)
# ============================================================

def recall_at_k(I, query_texts, ks):
    recalls = {k: 0 for k in ks}

    for i in range(N):
        q = embed_text(query_texts[i])
        sims = I @ q                    # cosine similarity
        rank = np.argsort(-sims)        # descending

        for k in ks:
            if i in rank[:k]:
                recalls[k] += 1

    for k in recalls:
        recalls[k] /= N

    return recalls

# ============================================================
# 5. Query 타입별 비교 실험
# ============================================================

results = {}

# ① Keyword baseline
keyword_queries = [KEYWORD_QUERIES[c] for c in classes]
results["Keyword"] = recall_at_k(I, keyword_queries, KS)

# ② Human-written sentence
human_queries = [HUMAN_SENTENCE_QUERIES[c] for c in classes]
results["Human Sentence"] = recall_at_k(I, human_queries, KS)

# ③ Auto-caption (BLIP)
auto_caption_queries = captions.tolist()
results["Auto Caption (BLIP)"] = recall_at_k(I, auto_caption_queries, KS)

# ============================================================
# 6. 결과 출력 (논문용)
# ============================================================

print("\n=== Text → Image Recall@K Comparison ===")
for method, res in results.items():
    print(f"\n[{method}]")
    for k in KS:
        print(f"Recall@{k}: {res[k]:.4f}")
