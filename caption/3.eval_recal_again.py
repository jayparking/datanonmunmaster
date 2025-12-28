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

KS = [1, 5, 10]

# ===============================
# 1. DB에서 embedding 로드
# ===============================
conn = psycopg2.connect(**DB_CONFIG)
register_vector(conn)
cur = conn.cursor()

cur.execute("""
    SELECT id, embedding, image_embedding
    FROM military_images
    WHERE embedding IS NOT NULL
      AND image_embedding IS NOT NULL
    ORDER BY id;
""")
rows = cur.fetchall()
cur.close()
conn.close()

ids = []
T = []  # text embeddings
I = []  # image embeddings

for _id, t, im in rows:
    ids.append(_id)
    T.append(t.astype(np.float32))
    I.append(im.astype(np.float32))

ids = np.array(ids)
T = np.stack(T)   # (N, 512)
I = np.stack(I)   # (N, 512)

N = T.shape[0]
print(f"Loaded {N} text-image pairs")

# ===============================
# 2. Text → Image Recall@K
# ===============================
def recall_at_k_text_to_image(T, I, ks):
    recalls = {k: 0 for k in ks}

    for i in range(N):
        # caption i → 모든 이미지 similarity
        sims = I @ T[i]        # (N,)
        rank = np.argsort(-sims)

        for k in ks:
            if i in rank[:k]:  # GT image가 top-k 안에 있으면 hit
                recalls[k] += 1

    for k in recalls:
        recalls[k] /= N

    return recalls

results = recall_at_k_text_to_image(T, I, KS)

# ===============================
# 3. 결과 출력
# ===============================
print("\n=== Text → Image Retrieval (GT pair) ===")
for k in KS:
    print(f"Recall@{k}: {results[k]:.4f}")
