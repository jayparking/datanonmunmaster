import glob
import json
from openai import OpenAI

# 🔑 OpenAI API Key 설정
client = OpenAI(api_key="오픈에이피")

# 📂 입력 캡션 JSON 파일들 (N개 자동 인식)
INPUT_JSON_FILES = glob.glob("../output2/web_image_captions_*.json")

print(f"🔍 Found {len(INPUT_JSON_FILES)} caption files")
if len(INPUT_JSON_FILES) == 0:
    print("❌ No input files found. Check path or filename pattern.")
    exit()


# ================================
#  GPT 태깅 함수
# ================================
def generate_tags(caption: str):
    """캡션을 입력받아 군사용 태그를 생성하는 함수"""

    prompt = f"""
    아래 캡션에 대해 군사용 태그를 3~5개 생성해줘.
    규칙:
    - 태그는 명사 기반 영어 단어
    - missile / rocket / aircraft 구분
    - 상황태그: launch-event, static-display 등 허용
    - 중복 금지
    - JSON 형식: {{"tags": ["...", "..."]}}

    Caption: "{caption}"
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    text = resp.choices[0].message.content.strip()

    # JSON 파싱 안전하게 처리
    try:
        parsed = json.loads(text)
        tags = parsed.get("tags", [])
    except json.JSONDecodeError:
        print("⚠️ JSON 파싱 실패. Raw response:", text)
        # fallback: 단순 쉼표 분리
        tags = [t.strip() for t in text.split(",") if t.strip()]

    return tags


# ================================
#  메인 루프: 모든 JSON 파일 처리
# ================================
for input_path in INPUT_JSON_FILES:

    # 파일명 내 captions → tagged 로 변경
    output_path = input_path.replace("captions", "tagged")

    print(f"\n🚀 Processing file: {input_path}")

    # 1) JSON 읽기
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    # 2) 각 데이터에 대해 태깅
    for item in data:
        caption = item.get("caption", "")
        tags = generate_tags(caption)

        # 태그 추가
        item["tags"] = tags
        results.append(item)

    # 3) 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved tagged file → {output_path}")


print("\n🎉 All caption files processed successfully!")
