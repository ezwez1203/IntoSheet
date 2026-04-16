# Copyright 2025 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert MIDI files into PDF sheet music via music21."""

import argparse
import os
from typing import List, Optional


def _find_musescore_path() -> Optional[str]:
  """Returns a likely MuseScore binary path if one exists."""
  common_paths = [
      r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
      r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
      r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe",
      r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
      "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
      "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
      "/usr/bin/musescore4",
      "/usr/bin/musescore3",
      "/usr/bin/musescore",
      "/usr/local/bin/musescore",
  ]
  for path in common_paths:
    if os.path.isfile(path):
      return path
  return None


def _configure_music21() -> Optional[str]:
  """Configures music21 to use an installed PDF renderer."""
  import music21  # pylint: disable=g-import-not-at-top
  import shutil  # pylint: disable=g-import-not-at-top

  env = music21.environment.Environment()

  mscore_path = _find_musescore_path()
  if mscore_path:
    env["musicxmlPath"] = mscore_path
    env["musescoreDirectPNGPath"] = mscore_path
    print(f"[midi_to_pdf] MuseScore found: {mscore_path}")
    return "musescore"

  lilypond_path = env.get("lilypondPath")
  if lilypond_path and os.path.isfile(str(lilypond_path)):
    print(f"[midi_to_pdf] LilyPond found: {lilypond_path}")
    return "lilypond"

  mscore_cmd = (
      shutil.which("musescore4")
      or shutil.which("musescore3")
      or shutil.which("musescore")
      or shutil.which("mscore")
  )
  if mscore_cmd:
    env["musicxmlPath"] = mscore_cmd
    env["musescoreDirectPNGPath"] = mscore_cmd
    print(f"[midi_to_pdf] MuseScore found in PATH: {mscore_cmd}")
    return "musescore"

  lilypond_cmd = shutil.which("lilypond")
  if lilypond_cmd:
    env["lilypondPath"] = lilypond_cmd
    print(f"[midi_to_pdf] LilyPond found in PATH: {lilypond_cmd}")
    return "lilypond"

  return None


def convert_midi_to_pdf(
    midi_path: str,
    output_dir: str = "./checkpoints/pdf_output",
    title: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> str:
  """Converts a single MIDI file to a PDF score."""
  if not os.path.isfile(midi_path):
    raise FileNotFoundError(f"MIDI file not found: {midi_path}")

  try:
    import music21  # pylint: disable=g-import-not-at-top
  except ImportError as exc:
    raise ImportError(
        "music21 is required for MIDI to PDF conversion. "
        "Install it with: pip install music21"
    ) from exc

  renderer = _configure_music21()
  if renderer is None:
    raise RuntimeError(
        "No PDF renderer found. Install MuseScore or LilyPond first."
    )

  os.makedirs(output_dir, exist_ok=True)

  print(f"[midi_to_pdf] Loading MIDI: {midi_path}")
  try:
    score = music21.converter.parse(midi_path)
  except Exception as exc:  # pylint: disable=broad-except
    raise ValueError(f"Failed to parse MIDI file '{midi_path}': {exc}") from exc

  if title:
    score.metadata = music21.metadata.Metadata()
    score.metadata.title = title

  if output_filename is None:
    basename = os.path.splitext(os.path.basename(midi_path))[0]
    output_filename = f"{basename}.pdf"
  elif not output_filename.endswith(".pdf"):
    output_filename += ".pdf"

  output_path = os.path.join(output_dir, output_filename)

  print(f"[midi_to_pdf] Rendering PDF with {renderer}...")
  try:
    if renderer == "lilypond":
      converter = music21.converter.subConverters.ConverterLilypond()
      converter.write(score, fmt="lilypond", fp=output_path, subformats=["pdf"])
    else:
      score.write("musicxml.pdf", fp=output_path)
  except Exception as exc:  # pylint: disable=broad-except
    raise RuntimeError(
        "Failed to render PDF. Verify MuseScore or LilyPond installation.\n"
        f"Error: {exc}"
    ) from exc

  print(f"[midi_to_pdf] PDF saved: {output_path}")
  return output_path


def convert_midi_directory(
    input_dir: str, output_dir: str = "./checkpoints/pdf_output"
) -> List[str]:
  """Converts all MIDI files in a directory to PDF."""
  midi_extensions = {".mid", ".midi", ".MID", ".MIDI"}
  pdf_paths = []

  for filename in sorted(os.listdir(input_dir)):
    if os.path.splitext(filename)[1] not in midi_extensions:
      continue
    midi_path = os.path.join(input_dir, filename)
    try:
      pdf_paths.append(convert_midi_to_pdf(midi_path, output_dir))
    except Exception as exc:  # pylint: disable=broad-except
      print(f"[midi_to_pdf] Skipping {filename}: {exc}")

  print(f"[midi_to_pdf] Converted {len(pdf_paths)} file(s).")
  return pdf_paths


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Convert MIDI files to PDF sheet music."
  )
  parser.add_argument("midi_path", nargs="?", help="Input MIDI file path.")
  parser.add_argument(
      "-o",
      "--output_dir",
      default="./checkpoints/pdf_output",
      help="Directory for generated PDFs.",
  )
  parser.add_argument("--title", default=None, help="Optional score title.")
  parser.add_argument(
      "--output_filename",
      default=None,
      help="Optional custom PDF filename.",
  )
  parser.add_argument(
      "--batch",
      default=None,
      help="Directory containing MIDI files to convert in batch.",
  )
  args = parser.parse_args()

  if args.batch:
    convert_midi_directory(args.batch, args.output_dir)
    return

  if not args.midi_path:
    parser.error("Either midi_path or --batch must be provided.")

  convert_midi_to_pdf(
      args.midi_path,
      output_dir=args.output_dir,
      title=args.title,
      output_filename=args.output_filename,
  )


if __name__ == "__main__":
  main()
