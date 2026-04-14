# 🚀 아주 작은 모두의 AI (TinyModooAI)

> **"누구나 자신의 컴퓨터에서 나만의 AI 모델을 만들고, 깎고, 서빙할 수 있도록"**

TinyModooAI는 Mistral 아키텍처를 기반으로 한 **초소형 언어 모델(SLM)**의 전체 생명 주기(Training to Serving)를 관리하는 고성능 파이프라인입니다. 현재 이 프로젝트는 파이프라인의 **동작 검증과 빠른 실험(Iteration)**을 위해 전체 데이터의 약 **1% 샘플**만을 사용하여 POC(Proof of Concept) 수준으로 구성되어 있습니다.

---

## 🏗️ 1. 아키텍처 및 철학
본 프로젝트는 vLLM 서빙이 가능한 표준 아키텍처를 채택하면서도, 로컬(Mac)에서의 효율을 극대화하도록 설계되었습니다.
- **Base Model**: Mistral-7B 구조를 극도로 축소 (8 Layers, 256 Hidden Size)
- **Attention**: GQA(Grouped-Query Attention) 적용으로 추론 효율성 증대
- **Strategy**: 하드웨어 제약 없이 누구나 모델의 밑바닥부터 학습 과정을 경험하는 것을 목표로 함

---

## 📊 2. 데이터셋 구성 및 학습 전략

학습은 두 단계(Pre-training ➡️ SFT)로 진행되며, 각 단계에서 사용되는 데이터셋의 역할은 다음과 같습니다.

### Phase 1: 사전 학습 (Pre-training)
지식을 습득하고 언어의 통계적 패턴을 학습하는 단계입니다.
- **데이터셋**: `Wikitext` (영어), `Wiki-Korean` (한국어)
- **포맷**: 비정형 텍스트 (Plain Text)
- **과업**: Causal Language Modeling (다음 단어 예측)
- **학습 방식**: 두 언어의 균형을 위해 샘플링 비율을 조정하여 혼합 학습 진행

### Phase 2: 지시어 튜닝 (SFT - Supervised Fine-Tuning)
학습된 지식을 바탕으로 사용자의 질문에 답변하는 '대화 지능'을 부여하는 단계입니다.
- **데이터셋 혼합**:
  1. **General**: `Alpaca` (지시어 처리 능력)
  2. **Coding**: `CodeAlpaca` (코드 생성 및 논리)
  3. **Math**: `GSM8K` (단계별 추론 능력)
- **포맷**: 지시어-입력-출력 (Instruction-Input-Response) 구조
- **학습 방식**: `### 사용자:`, `### 어시스턴트:` 프롬프트 템플릿을 적용하고, 모델이 어시스턴트의 답변 부분에 대해서만 손실(Loss)을 계산하도록 **Label Masking** 기법 적용

---

## ⚙️ 3. 파이프라인 단계별 과업 상세

### Step 1: 학습 (`1_train/`)
- **과업**: 모델의 뇌를 형성합니다.
- **디테일**: Mac MPS 가속 환경에 최적화된 `Accelerator` 설정을 사용하며, 현재는 **1% 샘플링** 모드로 설정되어 있습니다. (성능을 높이려면 스크립트 내 `select_percentage`를 상향하세요.)

### Step 2: 포맷 변환 (`2_convert/`)
- **과업**: 학습된 원본 가중치를 추론 엔진 규격에 맞춥니다.
- **디테일**: PyTorch의 `.bin` 가중치를 업계 표준인 `Safetensors`로 정밀하게 변환하고, vLLM이 인식할 수 있는 전용 설정 파일들을 생성합니다.

### Step 3: 경량화 (`3_quantize/`)
- **과업**: 모델의 무게를 줄이고 속도를 높입니다. (Diet for AI)
- **GGUF**: Mac 로컬 추론을 위한 Apple Silicon 가속 최적화 (4-bit/8-bit 선택 가능)
- **AWQ**: NVIDIA GPU 서버에서 vLLM의 성능을 극대화하기 위한 4-bit 양자화

### Step 4: 서빙 (`4_serve/`)
- **과업**: 완성된 AI를 외부(사용자)와 연결합니다.
- **하이브리드 지원**: Mac에서는 Transformers 기반 서버로, GPU 서버에서는 Docker/vLLM 기반으로 유연하게 배포 가능합니다.

---

## 🚀 빠른 시작 가이드 (Quick Start)

### 1단계: 학습
```bash
cd 1_train
python3 train.py      # 사전 학습 (Phase 1)
python3 train_sft.py  # 지시어 튜닝 (Phase 2)
```

### 2단계: 양자화 및 서빙
```bash
cd 3_quantize && python3 quantize_gguf.py  # Mac용 경량화
cd ../4_serve && ./run_vllm.sh             # 로컬 서빙 시작
```

---

## 🛠️ 기술 스택 (Tech Stack)
- **Model**: PyTorch, Transformers, Accelerate
- **Quantization**: llama.cpp (GGUF), AutoAWQ (AWQ)
- **Serving**: vLLM, FastAPI, Docker
- **Infras**: Apple Silicon (Local) & NVIDIA GPU (Cloud)

---
*Developed by Antigravity AI Assistant.*
