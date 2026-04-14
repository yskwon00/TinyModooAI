# 🚀 아주 작은 모두의 AI (TinyModooAI)

TinyModooAI는 vLLM 서빙이 가능한 표준화된 아키텍처(Mistral 기반)를 사용하는 고성능 소형 언어 모델(SLM) 학습 및 배포 파이프라인입니다.

## 🌟 주요 특징
- **표준 아키텍처**: Mistral 기반의 GQA(Grouped-Query Attention) 적용
- **하이트리드 학습**: 한국어(Wiki)와 영어(Wikitext) 혼합 사전 학습 지원
- **지시어 튜닝(SFT)**: Alpaca, Code, Reasoning 데이터를 혼합한 SFT 파이프라인
- **유연한 서빙**: Azure GPU 환경용 vLLM 도커와 Mac 로컬용 OpenAI 호환 서버 지원

## 🏗️ 프로젝트 구조 (Workflow 순서)
1. `1_train/`: 사전 학습 및 지시어 튜닝 (Base & SFT)
2. `2_convert/`: 학습된 가중치를 vLLM 최적화 포맷으로 변환
3. `3_quantize/`: 모델 용량 압축 및 가속 (GGUF & AWQ)
4. `4_serve/`: 로컬/클라우드 범용 API 서빙

---

## 🚀 빠른 시작 가이드

### 단계 1. 학습 (Training)
```bash
cd 1_train
python3 train.py      # 사전 학습 -> outputs/pretrained
python3 train_sft.py  # 지시어 튜닝 -> outputs/sft
```

### 단계 2. 변환 (Conversion)
```bash
cd 2_convert
python3 convert.py    # vLLM용 포맷으로 변환 -> outputs/vllm_*
```

### 단계 3. 경량화 (Quantization)
```bash
cd 3_quantize
python3 quantize_gguf.py  # GGUF 변환 (Mac 로컬 최적화)
python3 quantize_awq.py   # AWQ 변환 (GPU 가속 최적화)
```

### 단계 4. 서빙 (Serving)
```bash
cd 4_serve
./run_vllm.sh         # 서빙 모드 선택 (Local vs Docker)
```

---

## 🎨 활용 가이드 (Ollama & Open WebUI)

압축된 `.gguf` 모델을 **Ollama**에 등록하여 **Open WebUI**에서 사용하는 방법입니다.

### 1. Ollama 모델 등록
프로젝트 루트(최상위 폴더)에서 `Modelfile`이라는 이름의 파일을 만들고 아래 내용을 입력합니다.
```dockerfile
# 모델 파일 경로 (프로젝트 루트 기준)
FROM ./outputs/quantized/gguf/tinymodoo-sft-q4_k_m.gguf

# 대화 템플릿 설정
TEMPLATE """{{ .System }}
### 사용자:
{{ .Prompt }}
### 어시스턴트:
"""
PARAMETER stop "###"
```
그 후 터미널(루트 폴더)에서 모델을 생성합니다:
```bash
ollama create tinymodoo -f Modelfile
```

### 2. Open WebUI에서 대화하기
- Open WebUI를 실행하면 모델 목록에 `tinymodoo:latest`가 자동으로 나타납니다.
- 선택 후 대화를 시작하세요!

---

## 🛠️ 기술 스택
- **Core**: Python, PyTorch, Transformers, Datasets, Accelerate
- **Quantization**: llama.cpp, AutoAWQ
- **Serving**: vLLM, FastAPI, Docker
- **Hardware**: Apple Silicon (Mac) & NVIDIA GPU (Azure)

---
Developed by Antigravity AI Assistant.
