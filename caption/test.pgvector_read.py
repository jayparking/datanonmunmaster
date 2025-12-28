import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

DB_CONFIG = {
    "dbname": "mydatabase",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5433
}

def main():
    # 1. DB 연결
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)          # ⭐ 핵심
    cur = conn.cursor()

    # 2. 하나만 가져오기
    cur.execute("""
        SELECT embedding, image_embedding
        FROM military_images
        LIMIT 1;
    """)
    row = cur.fetchone()

    text_emb = row[0]
    image_emb = row[1]

    # 3. 타입 확인
    print("=== TEXT EMBEDDING ===")
    print("type:", type(text_emb))
    print("length:", len(text_emb))
    print("first 5:", text_emb[:5])

    print("\n=== IMAGE EMBEDDING ===")
    print("type:", type(image_emb))
    print("length:", len(image_emb))
    print("first 5:", image_emb[:5])

    # 4. numpy 변환 테스트
    text_np = np.array(text_emb, dtype=np.float32)
    image_np = np.array(image_emb, dtype=np.float32)

    print("\n✅ NumPy conversion success")
    print("text_np shape:", text_np.shape)
    print("image_np shape:", image_np.shape)

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
