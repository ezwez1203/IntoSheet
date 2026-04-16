# IntoSheet

🎵 **MT3 기반 음악 채보(Music Transcription) 시스템**

음원 파일(MP3/WAV)을 넣으면 딥러닝 모델이 자동으로 악보(MIDI)를 생성합니다.

---

📖 **문서 / Documentation**

- 🇰🇷 [한국어 README](README_KR.md)
- 🇺🇸 [English README](README_EN.md)

## Install

Install from the repository root only:

```bash
pip install -e .
```

Do not run `pip install -e mt3/` by itself. The local `mt3` package depends on
the local `t5x` package in this repository, so standalone `mt3/` installation
will fail on a fresh machine.
