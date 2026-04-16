# 🎵 IntoSheet — MT3 기반 음악 채보 시스템

> 음원 파일(MP3/WAV)을 넣으면 **딥러닝(MT3 Transformer)** 이 자동으로 악보(MIDI)를 만들어주는 프로젝트입니다.

---

## 프로젝트 구조

```
IntoSheet/
├── Partitur/               # 메인 응용 프로그램 (실행 진입점)
│   ├── run.py              # 실행 스크립트
│   ├── main.py             # Main 클래스 (MT3 래퍼)
│   └── mt3_inference.py    # MT3 InferenceModel 래퍼
├── mt3/                    # Google Magenta MT3 라이브러리
│   └── mt3/
│       ├── gin/            # 모델 설정 파일 (Gin configs)
│       ├── inference.py    # MT3 공식 추론 코드
│       ├── midi_to_pdf.py  # MIDI -> PDF 악보 변환
│       ├── spectrograms.py # 스펙트로그램 처리
│       └── ...
├── checkpoints/
│   ├── mt3/                # 사전 학습된 모델 체크포인트
│   └── pdf_output/         # PDF 악보 출력 디렉터리
├── t5x/                    # T5X 프레임워크 (학습용)
├── run_all.sh              # 원클릭 실행 스크립트
├── README_KR.md            # 이 파일 (한국어)
└── README_EN.md            # 영문 README
```

---

## 작동 원리

### Before (이전) — FFT 기반

기존 Partitur는 **푸리에 변환(FFT)** 으로 음파의 주파수를 분석한 뒤, 수식(`12 × log₂(f/440)`)으로 음높이를 추정했습니다. 단일 악기의 단순한 멜로디만 인식 가능했고, 화음이나 다중 악기에 대해서는 정확도가 매우 낮았습니다.

### After (현재) — MT3 Transformer

현재는 Google Magenta의 **MT3 (Multi-Task Multitrack Music Transcription)** 모델을 사용합니다.

| 항목 | FFT (이전) | MT3 (현재) |
|------|-----------|-----------|
| 핵심 기술 | scipy FFT + 피크 검출 | T5X Transformer 딥러닝 |
| 다중 악기 | ❌ | ✅ (128개 MIDI 프로그램) |
| 드럼 인식 | ❌ | ✅ |
| 출력 형식 | 텍스트 음이름 | 음이름 + MIDI 파일 |
| 정확도 | 낮음 | 높음 |

MT3는 오디오를 mel-spectrogram으로 변환한 뒤, encoder-decoder Transformer가 MIDI 토큰 시퀀스를 생성합니다. 이 토큰을 디코딩하면 `NoteSequence`(음표 시퀀스)가 됩니다.

---

## 시작하기

### 사전 준비

- **Python 3.11**
- **CUDA** 지원 GPU (JAX가 GPU를 사용합니다)
- 사전 학습 체크포인트: `checkpoints/mt3/` (이미 포함)

### 의존성 설치

```bash
# MT3 패키지 설치
pip install -e mt3/

# (필요시) JAX GPU 버전 설치
pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 추가 의존성
pip install librosa note-seq pyfluidsynth nest-asyncio
```

### 실행

#### 방법 1: 원클릭 스크립트

```bash
bash run_all.sh /path/to/audio.wav
```

#### MIDI를 PDF 악보로 변환

```bash
bash run_all.sh pdf ./output/song.mid
python -m mt3.midi_to_pdf ./output/song.mid -o ./checkpoints/pdf_output
```

`music21`과 함께 MuseScore 또는 LilyPond가 설치되어 있으면 PDF 렌더링이 가능합니다.

#### 방법 2: 직접 실행

```bash
# 커맨드라인 인자
python Partitur/run.py /path/to/audio.wav

# 또는 대화형 프롬프트
python Partitur/run.py
```

### 출력 결과

1. **콘솔 출력**: 감지된 음표 그룹 (예: `[['C4', 'E4', 'G4'], ['D4'], ...]`)
2. **MIDI 파일**: 입력 파일명 기반으로 자동 저장 (예: `audio_transcribed.mid`)
3. **PDF 악보**: `checkpoints/pdf_output/`에 저장 가능
4. **상세 요약**: 각 음표의 시작/끝 시간, 프로그램 번호, 드럼 여부

---

## 주요 API

### `Main` 클래스 (`Partitur/main.py`)

```python
from main import Main

m = Main("song.wav")

# 음이름 그룹 리스트 (기존 Partitur 호환)
notes = m.noteNames()
# [['C4', 'E4', 'G4'], ['D4', 'F#4'], ...]

# MIDI 파일 저장
m.save_midi("output.mid")

# 상세 요약 출력
m.print_summary()
```

### `InferenceModel` 클래스 (`Partitur/mt3_inference.py`)

```python
from mt3_inference import InferenceModel

model = InferenceModel()

# NoteSequence 반환
ns = model.transcribe("song.wav")

# dict 리스트 반환
notes = model.transcribe_to_notes("song.wav")
# [{'name': 'C4', 'pitch': 60, 'start': 0.0, 'end': 0.5, ...}, ...]

# MIDI 파일로 저장
model.transcribe_to_midi("song.wav", "output.mid")
```

---

## 모델 학습 (선택 사항)

기본적으로 사전 학습된 체크포인트(`checkpoints/mt3/`)를 사용하므로 별도 학습이 필요 없습니다. 커스텀 데이터셋으로 추가 학습(fine-tuning)이 필요한 경우:

1. `mt3/mt3/tasks.py`에 정의된 task를 확인합니다
2. [T5X 학습 가이드](https://github.com/google-research/t5x#training)를 따릅니다
3. `t5x/train.py`에 MT3 task를 연결하여 학습합니다

학습 중에는 터미널에 epoch 단위 progress bar가 표시되며, 가능한 경우 현재 `loss`와 최신 `F1` 점수가 함께 갱신됩니다.

---

## 참고 자료

- [MT3 논문 (ICLR 2022)](https://openreview.net/pdf?id=iMSjopcOn0p)
- [MT3 GitHub](https://github.com/magenta/mt3)
- [T5X 프레임워크](https://github.com/google-research/t5x)
- [MT3 Colab 노트북](https://colab.research.google.com/github/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb)

---

## 라이선스

- MT3: Apache License 2.0
- T5X: Apache License 2.0
- Partitur: 원본 프로젝트 라이선스 참조
