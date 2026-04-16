# 🎵 IntoSheet — MT3-Powered Music Transcription

> Feed in an audio file (MP3/WAV) and a **deep-learning Transformer (MT3)** automatically produces sheet music (MIDI).

---

## Project Structure

```
IntoSheet/
├── Partitur/               # Main application (entry point)
│   ├── run.py              # Runner script
│   ├── main.py             # Main class (MT3 wrapper)
│   └── mt3_inference.py    # MT3 InferenceModel wrapper
├── mt3/                    # Google Magenta MT3 library
│   └── mt3/
│       ├── gin/            # Model configuration (Gin configs)
│       ├── inference.py    # Official MT3 inference code
│       ├── midi_to_pdf.py  # MIDI -> PDF score conversion
│       ├── spectrograms.py # Spectrogram processing
│       └── ...
├── checkpoints/
│   ├── mt3/                # Pre-trained model checkpoint
│   └── pdf_output/         # PDF score output directory
├── t5x/                    # T5X framework (for training)
├── run_all.sh              # One-click run script
├── README_KR.md            # Korean README
└── README_EN.md            # This file (English)
```

---

## How It Works

### Before — FFT-Based

The original Partitur used **Fast Fourier Transform (FFT)** to analyse frequency peaks from audio waveforms and a formula (`12 × log₂(f/440)`) to estimate pitch. It could only handle simple monophonic melodies with a single instrument, and accuracy dropped sharply for chords or multi-instrument audio.

### After — MT3 Transformer

The project now uses Google Magenta's **MT3 (Multi-Task Multitrack Music Transcription)** model.

| Feature | FFT (Before) | MT3 (Now) |
|---------|-------------|-----------|
| Core Technology | scipy FFT + peak detection | T5X Transformer deep learning |
| Multi-instrument | ❌ | ✅ (128 MIDI programs) |
| Drum Recognition | ❌ | ✅ |
| Output Format | Text note names | Note names + MIDI file |
| Accuracy | Low | High |

MT3 converts audio into mel-spectrograms, then an encoder-decoder Transformer generates a sequence of MIDI tokens. Decoding these tokens yields a `NoteSequence` (a structured representation of musical notes).

---

## Getting Started

### Prerequisites

- **Python 3.11**
- **CUDA**-capable GPU (JAX uses GPU acceleration)
- Pre-trained checkpoint: `checkpoints/mt3/` (already included)

### Install Dependencies

```bash
# Recommended: install the whole project from the repository root
pip install -e .

# Or use the helper script
bash run_all.sh install
```

The root `setup.py` now installs:
- the pinned Python 3.11 external dependencies
- the extra Partitur runtime dependencies
- the local `t5x/` and `mt3/` packages

Do not run `pip install -e mt3/` by itself on a fresh machine.
`mt3/` depends on the local `t5x/` package in this same repository, and PyPI
does not provide a matching `t5x` package for that install path.

### Run

#### Option 1: One-Click Script

```bash
bash run_all.sh /path/to/audio.wav
```

#### Convert MIDI to PDF sheet music

```bash
bash run_all.sh pdf ./output/song.mid
python -m mt3.midi_to_pdf ./output/song.mid -o ./checkpoints/pdf_output
```

If `music21` and either MuseScore or LilyPond are installed, the project can
render generated MIDI files as PDF sheet music.

#### Option 2: Direct Execution

```bash
# Command-line argument
python Partitur/run.py /path/to/audio.wav

# Or interactive prompt
python Partitur/run.py
```

### Output

1. **Console**: Detected note groups (e.g. `[['C4', 'E4', 'G4'], ['D4'], ...]`)
2. **MIDI File**: Automatically saved based on input filename (e.g. `audio_transcribed.mid`)
3. **PDF Score**: Can be written into `checkpoints/pdf_output/`
4. **Summary**: Start/end time, program number, and drum flag for each note

---

## API Reference

### `Main` Class (`Partitur/main.py`)

```python
from main import Main

m = Main("song.wav")

# Note-name groups (backward-compatible with legacy Partitur)
notes = m.noteNames()
# [['C4', 'E4', 'G4'], ['D4', 'F#4'], ...]

# Save as MIDI
m.save_midi("output.mid")

# Print detailed summary
m.print_summary()
```

### `InferenceModel` Class (`Partitur/mt3_inference.py`)

```python
from mt3_inference import InferenceModel

model = InferenceModel()

# Returns a NoteSequence proto
ns = model.transcribe("song.wav")

# Returns a list of dicts
notes = model.transcribe_to_notes("song.wav")
# [{'name': 'C4', 'pitch': 60, 'start': 0.0, 'end': 0.5, ...}, ...]

# Export to MIDI file
model.transcribe_to_midi("song.wav", "output.mid")
```

---

## Training (Optional)

By default the pre-trained checkpoint (`checkpoints/mt3/`) is used, so no training is required. If you need to fine-tune on a custom dataset:

1. Review the tasks defined in `mt3/mt3/tasks.py`
2. Follow the [T5X training guide](https://github.com/google-research/t5x#training)
3. Connect the MT3 task to `t5x/train.py` and launch training

During training, the terminal now shows an epoch-scoped progress bar and
updates the current `loss` plus the most recent available `F1` score when
evaluation metrics exist.

---

## References

- [MT3 Paper (ICLR 2022)](https://openreview.net/pdf?id=iMSjopcOn0p)
- [MT3 GitHub](https://github.com/magenta/mt3)
- [T5X Framework](https://github.com/google-research/t5x)
- [MT3 Colab Notebook](https://colab.research.google.com/github/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb)

---

## License

- MT3: Apache License 2.0
- T5X: Apache License 2.0
- Partitur: See original project license
