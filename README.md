## ✅ 실험 파이프라인 요약 (Step-by-step)

### (A) 이미지 → Caption 생성 (BLIP)

- 군사 이미지 데이터셋 A의 n개 이미지를 **BLIP**에 입력하여 **이미지별 캡션**을 생성
    - 예: “A fighter jet flying in the sky”
    - 예: “A missile launcher vehicle on the ground”
- 생성된 **caption을 DB에 저장**

### (B) Caption → 검색 성능 기반 정량 검증 (CLIP)

- 저장된 caption을 **CLIP text encoder**로 임베딩(벡터화)
- DB에 저장된 **image embedding(CLIP)**과 caption embedding 간 **코사인 유사도**를 계산
- Text→Image retrieval 관점에서 **Recall@K(1,5,10)**로 성능을 측정
- 비교 기준(텍스트 입력 3종)
    1. Keyword (단순 키워드)
    2. Human Sentence (사람이 작성한 문장)
    3. Auto Caption (BLIP 캡션)

---

## ✅ 결과 (Text → Image Recall@K)

**[Keyword]**

- Recall@1: 0.0060
- Recall@5: 0.0298
- Recall@10: 0.0595

**[Human Sentence]**

- Recall@1: 0.0060
- Recall@5: 0.0298
- Recall@10: 0.0595

**[Auto Caption (BLIP)]**

- Recall@1: 0.1012
- Recall@5: 0.3095
- Recall@10: 0.4226

---

## ✅ 한 줄 결론

- **BLIP 기반 Auto-caption이 Keyword / Human Sentence 대비 Text→Image retrieval 성능(Recall@K)을 크게 향상**시켰다.
