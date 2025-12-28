import os
import json
from PIL import Image
import torch
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# 0. 설정 부분 ===========================
IMAGE_DIR = "../images/all"  # 이미지 폴더 경로
OUTPUT_JSON = "../output2/all.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Salesforce/blip-image-captioning-base"
# ======================================

def load_model():
    print(f"Loading BLIP model: {MODEL_NAME}")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return processor, model

def generate_caption(image, processor, model, max_new_tokens=30):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=3,          # 살짝 더 좋은 문장
            repetition_penalty=1.2
        )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def main():
    processor, model = load_model()

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    image_files.sort()

    results = []

    for idx, fname in enumerate(tqdm(image_files, desc="Captioning")):
        img_path = os.path.join(IMAGE_DIR, fname)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open {img_path}: {e}")
            continue

        caption = generate_caption(image, processor, model)

        # 나중에 태깅/학습에 쓰기 편하게 구조 설계
        results.append({
            "id": idx,                 # 이미지 ID
            "file_name": fname,        # 파일 이름
            "image_path": img_path,    # 절대/상대 경로
            "caption": caption         # 생성된 문장
        })

    # JSON 파일로 저장
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n>> Saved {len(results)} captions to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
