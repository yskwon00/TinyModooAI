# 🚀 아주 작은 모두의 AI (TinyModooAI)

TinyModooAI는 vLLM 서빙이 가능한 표준화된 아키텍처(Mistral 기반)를 사용하는 고성능 소형 언어 모델(SLM) 학습 및 배포 파이프라인입니다.

## 🌟 주요 특징
- **표준 아키텍처**: Mistral 기반의 GQA(Grouped-Query Attention) 적용
- **하이브리드 학습**: 한국어(Wiki)와 영어(Wikitext) 혼합 사전 학습 지원
- **지시어 튜닝(SFT)**: Alpaca, Code, Reasoning 데이터를 혼합한 SFT 파이프라인
- **유연한 서빙**: Azure GPU 환경용 vLLM 도커와 Mac 로컬용 OpenAI 호환 서버 동시 지원

## 🏗️ 프로젝트 구조
- `1_train/`: 사전 학습 및 지시어 튜닝 스크립트
- `2_convert/`: 학습된 가중치를 vLLM 최적화 포맷으로 변환
- `3_serve/`: 서빙 모듈 (Mac Local & Docker/vLLM)
- `4_quantize/`: 모델 경량화 (GGUF & AWQ)

## 🚀 빠른 시작 가이드

### 1. 학습 (Training)
```bash
cd 1_train
python3 train.py      # 사전 학습 -> outputs/pretrained
python3 train_sft.py  # 지시어 튜닝 -> outputs/sft
```

### 2. 경량화 (Quantization) - NEW!
```bash
cd 4_quantize
python3 quantize_gguf.py  # GGUF 변환 (4-bit/8-bit 선택 가능)
python3 quantize_awq.py   # AWQ 변환 (GPU 전용)
```

### 2. 변환 (Conversion)
```bash
cd 2_convert
python3 convert.py    # vLLM용 포맷으로 변환
```

### 3. 서빙 (Serving)
```bash
cd 3_serve
./run_vllm.sh         # 서빙 모드 선택 (Mac Local vs Docker)
```

## 🛠️ 기술 스택
- **Core**: Python, PyTorch, Transformers, Datasets, Accelerate
- **Serving**: vLLM, FastAPI, Docker
- **Hardware**: Apple Silicon (Mac) & NVIDIA GPU (Azure)

---
Developed by Antigravity AI Assistant.
