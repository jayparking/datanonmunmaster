import json
import psycopg2
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pgvector.psycopg2 import register_vector

# ================= 설정 =================
JSON_PATH = "../output2/all.json"

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

# 1. CLIP 로드
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
clip_model.eval()

# 2. 텍스트 임베딩 함수
def embed_text(text: str):
    inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)
    
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].tolist()

# 3. 이미지 임베딩 함수
def embed_image(image_path: str):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].tolist()

# 4. DB 연결
conn = psycopg2.connect(**DB_CONFIG)

register_vector(conn)   # 🔥 여기 딱 한 줄

cur = conn.cursor()


# 5. JSON 로드
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Processing {len(data)} items...")

# 6. Insert
try:
    for item in tqdm(data):
        caption = item["caption"]
        image_path = item["image_path"]
        
        text_embedding = embed_text(caption)
        image_embedding = embed_image(image_path)
        
        cur.execute(
            """
            INSERT INTO military_images (image_path, caption, embedding, image_embedding)
            VALUES (%s, %s, %s, %s)
            """,
            (image_path, caption, text_embedding, image_embedding)
        )
    
    conn.commit()
    print(f"✅ Done: {len(data)} items inserted into pgvector")
    
except Exception as e:
    conn.rollback()
    print(f"❌ Error: {e}")
    
finally:
    cur.close()
    conn.close()