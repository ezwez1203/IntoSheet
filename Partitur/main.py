"""
Partitur — Music Transcription powered by MT3.

This module used to extract notes from audio via FFT (Fourier Transform).
It has been rewritten to delegate transcription to the MT3 deep-learning
model, which produces far more accurate multi-instrument results.

Legacy classes (Extractor, Transformator, Translator) are no longer used;
the ``Main`` class now wraps ``mt3_inference.InferenceModel``.
"""

from mt3_inference import InferenceModel, midi_pitch_to_name


class Main:
    """High-level API for audio → note transcription using MT3.

    Usage::

        m = Main("song.wav")
        notes = m.get_notes()          # list of note dicts
        m.save_midi("output.mid")      # export MIDI
        names = m.noteNames()          # [['C4', 'E4', 'G4'], ...]
    """

    def __init__(self, file: str, model_type: str = "mt3"):
        self.file = file.strip('"')
        print(f"[Partitur] MT3 모델을 로딩합니다 (model_type={model_type})...")
        self.model = InferenceModel(model_type=model_type)
        print("[Partitur] 모델 로딩 완료.")
        self._notes = None
        self._ns = None

    # ------------------------------------------------------------------
    # core transcription
    # ------------------------------------------------------------------
    def transcribe(self):
        """Run MT3 inference and cache the results."""
        if self._ns is None:
            print(f"[Partitur] 오디오 분석 중: {self.file}")
            self._ns = self.model.transcribe(self.file)
            self._notes = self.model.transcribe_to_notes(self.file)
            print(f"[Partitur] 총 {len(self._notes)}개의 음표를 감지했습니다.")
        return self._ns

    def get_notes(self):
        """Return a list of note dicts (see ``InferenceModel.transcribe_to_notes``)."""
        self.transcribe()
        return self._notes

    # ------------------------------------------------------------------
    # output helpers — keep familiar Partitur-style interface
    # ------------------------------------------------------------------
    def noteNames(self):
        """Return detected notes grouped by approximate time windows.

        Time windows are 0.2 s wide (matching the old ``splitLengthInSeconds``).
        Within each window, simultaneous notes are grouped into a sub-list of
        note-name strings.

        Returns:
            ``[['C4', 'E4', 'G4'], ['D4'], ...]``
        """
        notes = self.get_notes()
        if not notes:
            return []

        window = 0.2  # seconds — same as the old split length
        groups: list[list[str]] = []
        current_group: list[str] = []
        current_window_start = notes[0]["start"]

        for note in notes:
            if note["start"] - current_window_start >= window:
                if current_group:
                    groups.append(current_group)
                current_group = [note["name"]]
                current_window_start = note["start"]
            else:
                current_group.append(note["name"])

        if current_group:
            groups.append(current_group)

        # deduplicate within each group (like the old removeRepetitions)
        deduped = []
        for group in groups:
            seen = set()
            unique = []
            for name in group:
                if name not in seen:
                    seen.add(name)
                    unique.append(name)
            deduped.append(unique)

        # remove consecutive identical groups
        result = [deduped[0]] if deduped else []
        for g in deduped[1:]:
            if sorted(g) != sorted(result[-1]):
                result.append(g)

        return result

    def save_midi(self, output_path: str = "transcribed.mid"):
        """Export the transcription result as a MIDI file."""
        self.model.transcribe_to_midi(self.file, output_path)

    # ------------------------------------------------------------------
    # pretty-print
    # ------------------------------------------------------------------
    def print_summary(self):
        """Print a readable summary of detected notes."""
        notes = self.get_notes()
        print(f"\n=== 감지된 음표: {len(notes)}개 ===")
        for i, n in enumerate(notes):
            drum = " [드럼]" if n["is_drum"] else ""
            print(
                f"  {i+1:4d}. {n['name']:>4s}  "
                f"시작={n['start']:7.3f}s  "
                f"끝={n['end']:7.3f}s  "
                f"프로그램={n['program']}{drum}"
            )
