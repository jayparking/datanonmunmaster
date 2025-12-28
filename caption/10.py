import numpy as np
import psycopg2
import torch
from transformers import CLIPProcessor, CLIPModel
from pgvector.psycopg2 import register_vector

# ================= 설정 =================
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "dbname": "mydatabase",
    "user": "postgres",
    "password": "postgres",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "openai/clip-vit-base-patch32"
KS = [1, 5]

TEXT_QUERIES = {
    "aircraft": [
        "fighter aircraft on runway",
        "military jet flying in the sky",
    ],
    "tank": [
        "main battle tank on ground",
        "armored tank vehicle",
    ],
    "missile": [
        "missile launcher vehicle",
        "ballistic missile system",
    ],
    "naval_ship": [
        "naval warship at sea",
        "military ship on ocean",
    ],
}
# =======================================

# ================= 클래스 정규화 (강화) =================
def normalize_class(raw: str) -> str:
    raw = raw.lower()

    if any(k in raw for k in [
        "aircraft", "jet", "fighter", "fixed-wing", "airplane"
    ]):
        return "aircraft"

    if any(k in raw for k in [
        "tank", "armored", "mbt"
    ]):
        return "tank"

    if any(k in raw for k in [
        "missile", "launcher", "tel", "sam", "rocket", "ballistic"
    ]):
        return "missile"

    if any(k in raw for k in [
        "ship", "naval", "destroyer", "frigate", "carrier"
    ]):
        return "naval_ship"

    return "other"

# ================= CLIP 로드 =================
print("Loading CLIP...")
model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
model.eval()

def embed_text(text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

# ================= DB 로드 =================
print("Loading embeddings from DB...")
conn = psycopg2.connect(**DB_CONFIG)
register_vector(conn)
cur = conn.cursor()

cur.execute("""
    SELECT image_path, embedding, image_embedding
    FROM military_images
""")
rows = cur.fetchall()
cur.close()
conn.close()

paths, T, I, classes = [], [], [], []

for path, t, im in rows:
    if t is None or im is None:
        continue

    raw_cls = path.replace("\\", "/").split("/")[-2]
    cls = normalize_class(raw_cls)

    paths.append(path)
    T.append(np.array(t, dtype=np.float32))
    I.append(np.array(im, dtype=np.float32))
    classes.append(cls)

T = np.stack(T)
I = np.stack(I)
classes = np.array(classes)

# ================= 클래스 분포 출력 (중요) =================
print("\nClass distribution:")
unique, counts = np.unique(classes, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c}")

# ================= 평가 함수 =================
def recall_at_k_text_query(
    query_emb: np.ndarray,
    db_emb: np.ndarray,
    gt_class: str,
    k: int,
    debug: bool = False,
) -> bool:
    sims = db_emb @ query_emb
    topk = np.argsort(-sims)[:k]

    if debug:
        print("\n[DEBUG]")
        print("GT class:", gt_class)
        print("Top-K classes:", classes[topk])
        print("Top-K paths:", [paths[i] for i in topk])

    return np.any(classes[topk] == gt_class)

# ================= 평가 실행 =================
results = {
    "image_only": {k: [] for k in KS},
    "caption": {k: [] for k in KS},
}

print("\n=== Text Query → Image Retrieval ===")

for gt_class, queries in TEXT_QUERIES.items():
    for q in queries:
        q_emb = embed_text(q)

        for k in KS:
            results["image_only"][k].append(
                recall_at_k_text_query(q_emb, I, gt_class, k)
            )
            results["caption"][k].append(
                recall_at_k_text_query(q_emb, T, gt_class, k)
            )

# ================= 결과 출력 =================
for k in KS:
    r_img = np.mean(results["image_only"][k])
    r_txt = np.mean(results["caption"][k])
    print(f"Recall@{k} | Image-only: {r_img:.3f} | Caption-based: {r_txt:.3f}")
