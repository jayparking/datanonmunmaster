import psycopg2
import traceback

try:
    print("연결 시도 중...")
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5433,
        dbname="mydatabase",
        user="postgres",
        password="postgres",
        connect_timeout=10
    )
    print("✅ 연결 성공!")
    
    cur = conn.cursor()
    cur.execute("SELECT version();")
    version = cur.fetchone()
    print(f"PostgreSQL 버전: {version[0]}")
    
    cur.close()
    conn.close()
    
except psycopg2.OperationalError as e:
    print(f"❌ 연결 실패 (OperationalError): {e}")
    traceback.print_exc()
except Exception as e:
    print(f"❌ 연결 실패: {e}")
    traceback.print_exc()