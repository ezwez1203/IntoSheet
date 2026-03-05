# 🎵 IntoSheet — 음악을 악보로 변환하는 AI 프로젝트

음악 파일(MP3, WAV 등)을 입력받아 AI 모델을 활용하여 자동으로 **MIDI 파일**로 변환하고, 이를 **PDF 악보**로 렌더링하는 프로젝트입니다.  
Google Magenta의 **MT3 (Multi-Task Multitrack Music Transcription)** 모델과 **T5X** 프레임워크를 기반으로 합니다.

---

## 📁 폴더 구조

```
IntoSheet/
├── mt3/                    # MT3 음악 전사(Transcription) 모델
│   ├── mt3/
│   │   ├── colab/          # Colab 노트북 (추론 데모)
│   │   │   ├── music_transcription_with_transformers.ipynb
│   │   │   └── mt3_reconvat_baseline.ipynb
│   │   ├── gin/            # Gin 설정 파일 (모델/학습/추론 구성)
│   │   │   ├── model.gin
│   │   │   ├── mt3.gin
│   │   │   ├── ismir2021.gin
│   │   │   ├── train.gin
│   │   │   ├── eval.gin
│   │   │   ├── infer.gin
│   │   │   └── local_tiny.gin
│   │   ├── scripts/        # 유틸리티 스크립트
│   │   ├── midi_to_pdf.py  # ★ MIDI → PDF 악보 변환 스크립트
│   │   ├── datasets.py     # 데이터셋 정의
│   │   ├── inference.py    # 추론 로직
│   │   ├── models.py       # 모델 정의
│   │   ├── network.py      # Transformer 네트워크 구조
│   │   ├── tasks.py        # 학습 태스크 정의
│   │   ├── preprocessors.py    # 데이터 전처리
│   │   ├── spectrograms.py     # 스펙트로그램 변환
│   │   ├── vocabularies.py     # 음악 이벤트 어휘 사전
│   │   ├── note_sequences.py   # 노트 시퀀스 인코딩/디코딩
│   │   └── ...
│   └── setup.py            # MT3 패키지 설치 스크립트
│
├── t5x/                    # T5X 학습 프레임워크 (JAX 기반)
│   ├── t5x/
│   │   ├── train.py        # 학습 실행 스크립트
│   │   ├── eval.py         # 평가 실행 스크립트
│   │   ├── infer.py        # 추론 실행 스크립트
│   │   ├── models.py       # T5X 모델 추상 클래스
│   │   ├── trainer.py      # 학습 루프
│   │   ├── decoding.py     # 디코딩 알고리즘
│   │   ├── checkpoints.py  # 체크포인트 관리
│   │   ├── configs/        # 기본 설정 파일들
│   │   └── ...
│   └── setup.py            # T5X 패키지 설치 스크립트
│
├── checkpoints/            # 사전학습된 모델 체크포인트 및 출력
│   ├── mt3/                # MT3 다악기 전사 모델 체크포인트 (8-layer Transformer)
│   ├── my_mt3_tiny/        # 커스텀 MT3 Tiny 모델 체크포인트
│   └── pdf_output/         # ★ MIDI → PDF 변환 결과 저장 폴더
│
├── requirements.txt        # 전체 프로젝트 의존성 (conda 설치 용)
└── README.md               # ← 이 파일
```

---

## 🔍 각 폴더별 기능 설명

### `mt3/` — MT3 음악 전사 모델

Google Magenta 팀이 개발한 **Transformer 기반 다악기 자동 음악 전사(Automatic Music Transcription)** 모델입니다.

- **핵심 기능**: 오디오 → 스펙트로그램 → Transformer 인코더-디코더 → MIDI 노트 시퀀스 → PDF 악보
- **지원 모델**:
  - `ismir2021` — 피아노 전용 전사 (벨로시티 포함)
  - `mt3` — 다악기 전사 (벨로시티 없음)
- **주요 파일**:
  - `midi_to_pdf.py` — **MIDI → PDF 악보 변환** (music21 + MuseScore/LilyPond)
  - `network.py` — T5 기반 Transformer 아키텍처 정의
  - `models.py` — 연속 입력 인코더-디코더 모델
  - `inference.py` — 추론 및 결과 기록
  - `tasks.py` — SeqIO 학습 태스크 정의
  - `spectrograms.py` — 오디오 → 멜 스펙트로그램 변환
  - `vocabularies.py` — 음악 이벤트를 토큰으로 인코딩하는 어휘 사전
  - `colab/music_transcription_with_transformers.ipynb` — 인터랙티브 전사 데모

### `t5x/` — T5X 학습 프레임워크

Google Research의 **JAX 기반 T5 모델 학습 프레임워크**입니다. MT3 모델의 학습, 평가, 추론 기반 인프라를 제공합니다.

- **핵심 기능**: 분산 학습, 체크포인트 관리, 평가/추론 파이프라인
- **주요 파일**:
  - `train.py` — 모델 학습 실행
  - `eval.py` — 모델 평가 실행
  - `infer.py` — 모델 추론 실행
  - `trainer.py` — 학습 루프 구현
  - `decoding.py` — 빔 서치 등 디코딩 알고리즘
  - `checkpoints.py` — 체크포인트 저장/복원

### `checkpoints/` — 모델 체크포인트 및 출력

- `mt3/` — 공식 MT3 다악기 전사 모델 체크포인트 (8-layer Encoder + 8-layer Decoder)
- `my_mt3_tiny/` — 커스텀 경량 MT3 모델 (학습 설정 `config.gin` 포함)
- `pdf_output/` — MIDI → PDF 변환 결과가 저장되는 폴더

---

## 🤖 AI 모델 학습 방법

### 방법 1: Google Colab에서 사전학습 모델 사용 (추천)

가장 간단한 방법은 Colab 노트북을 사용하는 것입니다:

1. `mt3/mt3/colab/music_transcription_with_transformers.ipynb` 노트북을 Google Colab에서 엽니다.
2. **Runtime > Change Runtime Type > GPU** 선택
3. 셀을 순서대로 실행합니다.

### 방법 2: 로컬에서 MT3 모델 학습

```bash
# 1. 가상환경 활성화 (아래 "가상환경 구축" 섹션 참조)
conda activate intosheet

# 2. MT3 패키지 설치
cd mt3
pip install -e .

# 3. T5X 패키지 설치
cd ../t5x
pip install -e .

# 4. T5X를 사용하여 학습 실행 (예: 로컬 Tiny 모델)
python -m t5x.train \
  --gin_file="mt3/mt3/gin/local_tiny.gin" \
  --gin.MODEL_DIR="'./checkpoints/my_mt3_tiny'" \
  --gin.MIXTURE_OR_TASK_NAME="'onsets_and_offsets'" \
  --alsologtostderr
```

> ⚠️ **참고**: MT3 학습은 대규모 데이터셋과 GPU/TPU 자원이 필요합니다. 로컬 학습은 `local_tiny.gin` 설정으로 소규모 테스트가 가능합니다.

---

## 🎼 음악 파일을 악보로 변환하는 방법

### Step 1: 음악 → MIDI 변환 (MT3 AI 모델)

#### Google Colab 사용 (추천)

1. `mt3/mt3/colab/music_transcription_with_transformers.ipynb`을 Colab에서 엽니다.
2. 모델 유형을 선택합니다:
   - `ismir2021` — 피아노 전용 (벨로시티 포함)
   - `mt3` — 다악기 (벨로시티 없음)
3. **Upload Audio** 셀에서 MP3 또는 WAV 파일을 업로드합니다.
4. **Transcribe Audio** 셀을 실행하면 MIDI 파일로 변환됩니다.
5. **Download MIDI Transcription** 셀에서 결과 MIDI를 다운로드합니다.

#### 로컬에서 사용

```bash
# 가상환경 활성화
conda activate intosheet

# 음악 파일을 프로젝트 루트에 위치시킵니다.
# 예: IntoSheet/my_music.wav

python -m t5x.infer \
  --gin_file="mt3/mt3/gin/infer.gin" \
  --gin.CHECKPOINT_PATH="'./checkpoints/mt3'" \
  --gin.INFER_OUTPUT_DIR="'./output'" \
  --gin.INPUT_FILE="'./my_music.wav'" \
  --alsologtostderr
```

### Step 2: MIDI → PDF 악보 변환

MT3로 생성한 MIDI 파일을 PDF 악보로 변환합니다.

> ⚠️ **사전 요구사항**: [MuseScore](https://musescore.org/download) 또는 [LilyPond](https://lilypond.org/download.html)가 시스템에 설치되어 있어야 합니다.

```bash
# 단일 MIDI 파일을 PDF로 변환
python -m mt3.midi_to_pdf my_song.mid

# 출력 폴더 및 제목 지정
python -m mt3.midi_to_pdf my_song.mid -o ./checkpoints/pdf_output --title "나의 노래"

# 디렉토리 내 모든 MIDI 파일을 일괄 변환
python -m mt3.midi_to_pdf --batch ./midi_folder -o ./checkpoints/pdf_output
```

변환된 PDF 파일은 기본적으로 `checkpoints/pdf_output/` 폴더에 저장됩니다.

#### Python에서 직접 사용

```python
from mt3.midi_to_pdf import convert_midi_to_pdf

# 단일 파일 변환
pdf_path = convert_midi_to_pdf("my_song.mid", output_dir="./checkpoints/pdf_output")

# 제목 포함
pdf_path = convert_midi_to_pdf("my_song.mid", title="나의 노래")
```

---

## 🐍 가상환경 구축

### Conda 가상환경 생성 및 패키지 설치

```bash
# 1. Conda 가상환경 생성 (Python 3.10 권장)
conda create -n intosheet python=3.10 -y

# 2. 가상환경 활성화
conda activate intosheet

# 3. requirements.txt를 사용하여 패키지 설치
pip install -r requirements.txt

# 4. MT3 패키지 설치 (개발 모드)
cd mt3
pip install -e .
cd ..

# 5. T5X 패키지 설치 (개발 모드)
cd t5x
pip install -e .
cd ..
```

### 가상환경 비활성화

```bash
conda deactivate
```

---

## 📋 참고 사항

- **GPU 필수**: MT3 모델의 학습 및 추론에는 NVIDIA GPU와 CUDA가 필요합니다.
- **Colab 추천**: 로컬 GPU가 없는 경우, Google Colab의 무료 GPU를 활용하세요.
- **ffmpeg 필요**: MP3 파일을 사용하려면 시스템에 [ffmpeg](https://ffmpeg.org/download.html)가 설치되어 있어야 합니다.
- **PDF 변환**: MIDI → PDF 변환에는 [MuseScore](https://musescore.org/download) 또는 [LilyPond](https://lilypond.org/download.html)가 필요합니다.
- **체크포인트 다운로드**: 공식 MT3 체크포인트는 `gsutil -m cp -r gs://mt3/checkpoints .` 명령어로 다운로드할 수 있습니다.
