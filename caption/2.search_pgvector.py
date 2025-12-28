import psycopg2
import torch
from transformers import CLIPProcessor, CLIPModel




# 쿼리부분에 원하는 값넣으면 됩니다
# ================= 설정 =================
QUERY = "fighter"
TOP_K = 5

DB_CONFIG = {
    "dbname": "mydatabase",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5433
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "openai/clip-vit-base-patch32"
# =======================================

# CLIP 로드
clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
clip_model.eval()

def embed_text(text):
    inputs = clip_processor(text=[text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].tolist()

query_emb = embed_text(QUERY)

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute(
    """
    SELECT image_path, caption
    FROM military_images
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """,
    (query_emb, TOP_K)
)


results = cur.fetchall()

print(f"\n🔎 Query: {QUERY}")
for i, (path, caption) in enumerate(results):
    print(f"{i+1}. {path}")
    print(f"   {caption}\n")

cur.close()
conn.close()
