"""
Partitur Runner — MT3 기반 음악 채보 실행 스크립트.

사용법:
    python run.py                       # 대화형 프롬프트
    python run.py /path/to/audio.wav    # 커맨드라인 인자
"""

import sys
import os

from main import Main


def run():
    # ---- 오디오 파일 경로 가져오기 ----
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        file = input("파일 경로 (mp3/wav): ").strip()

    if not os.path.isfile(file):
        print(f"[오류] 파일을 찾을 수 없습니다: {file}")
        sys.exit(1)

    # ---- MT3 모델로 채보 ----
    m = Main(file)
    notes = m.noteNames()

    print(f"\n감지된 음표 그룹 수: {len(notes)}")
    print(notes)

    # ---- MIDI 파일 저장 여부 ----
    midi_path = os.path.splitext(file)[0] + "_transcribed.mid"
    m.save_midi(midi_path)

    # ---- 상세 요약 출력 ----
    m.print_summary()


if __name__ == "__main__":
    run()
