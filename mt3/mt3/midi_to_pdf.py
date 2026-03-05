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

"""MIDI to PDF sheet music converter using music21.

This module converts MIDI files to PDF sheet music using the music21 library.
It requires either MuseScore or LilyPond to be installed on the system for
PDF rendering.

Usage (CLI):
    python -m mt3.midi_to_pdf input.mid --output_dir ./checkpoints/pdf_output
    python -m mt3.midi_to_pdf input.mid -o ./output --title "My Song"

Usage (Python):
    from mt3.midi_to_pdf import convert_midi_to_pdf
    pdf_path = convert_midi_to_pdf("input.mid", output_dir="./output")
"""

import argparse
import os
import sys
from typing import Optional


def _find_musescore_path() -> Optional[str]:
    """Try to find MuseScore installation path on the system."""
    common_paths = [
        # Windows
        r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
        r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
        r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe",
        r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
        # macOS
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
        "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
        # Linux
        "/usr/bin/musescore4",
        "/usr/bin/musescore3",
        "/usr/bin/musescore",
        "/usr/local/bin/musescore",
    ]
    for path in common_paths:
        if os.path.isfile(path):
            return path
    return None


def _configure_music21():
    """Configure music21 to use available PDF renderer."""
    import music21  # pylint: disable=g-import-not-at-top

    env = music21.environment.Environment()

    # Try MuseScore first
    mscore_path = _find_musescore_path()
    if mscore_path:
        env['musicxmlPath'] = mscore_path
        env['musescoreDirectPNGPath'] = mscore_path
        print(f"[midi_to_pdf] MuseScore found: {mscore_path}")
        return 'musescore'

    # Check if LilyPond is available
    lilypond_path = env.get('lilypondPath')
    if lilypond_path and os.path.isfile(str(lilypond_path)):
        print(f"[midi_to_pdf] LilyPond found: {lilypond_path}")
        return 'lilypond'

    # Try system PATH
    import shutil  # pylint: disable=g-import-not-at-top
    mscore_cmd = shutil.which('musescore4') or shutil.which('musescore3') or shutil.which('musescore') or shutil.which('mscore')
    if mscore_cmd:
        env['musicxmlPath'] = mscore_cmd
        env['musescoreDirectPNGPath'] = mscore_cmd
        print(f"[midi_to_pdf] MuseScore found in PATH: {mscore_cmd}")
        return 'musescore'

    lilypond_cmd = shutil.which('lilypond')
    if lilypond_cmd:
        env['lilypondPath'] = lilypond_cmd
        print(f"[midi_to_pdf] LilyPond found in PATH: {lilypond_cmd}")
        return 'lilypond'

    return None


def convert_midi_to_pdf(
    midi_path: str,
    output_dir: str = './checkpoints/pdf_output',
    title: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> str:
    """Convert a MIDI file to PDF sheet music.

    Args:
        midi_path: Path to the input MIDI file.
        output_dir: Directory to save the output PDF. Defaults to
            './checkpoints/pdf_output'.
        title: Optional title to display on the sheet music.
        output_filename: Optional custom filename for the output PDF.
            If not provided, uses the MIDI filename with .pdf extension.

    Returns:
        Path to the generated PDF file.

    Raises:
        FileNotFoundError: If the MIDI file does not exist.
        RuntimeError: If no PDF renderer (MuseScore/LilyPond) is found.
        ValueError: If the MIDI file cannot be parsed.
    """
    # Validate input
    if not os.path.isfile(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    # Import music21 (heavy import, do it lazily)
    try:
        import music21  # pylint: disable=g-import-not-at-top
    except ImportError:
        raise ImportError(
            "music21 is required for MIDI to PDF conversion. "
            "Install it with: pip install music21"
        )

    # Configure renderer
    renderer = _configure_music21()
    if renderer is None:
        raise RuntimeError(
            "No PDF renderer found. Please install one of the following:\n"
            "  - MuseScore: https://musescore.org/download\n"
            "  - LilyPond: https://lilypond.org/download.html\n"
            "After installation, restart your terminal/IDE."
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Parse MIDI file
    print(f"[midi_to_pdf] Loading MIDI: {midi_path}")
    try:
        score = music21.converter.parse(midi_path)
    except Exception as e:
        raise ValueError(f"Failed to parse MIDI file '{midi_path}': {e}")

    # Set title if provided
    if title:
        score.metadata = music21.metadata.Metadata()
        score.metadata.title = title

    # Determine output path
    if output_filename is None:
        basename = os.path.splitext(os.path.basename(midi_path))[0]
        output_filename = f"{basename}.pdf"

    if not output_filename.endswith('.pdf'):
        output_filename += '.pdf'

    output_path = os.path.join(output_dir, output_filename)

    # Convert to PDF
    print(f"[midi_to_pdf] Rendering PDF with {renderer}...")
    try:
        if renderer == 'lilypond':
            # Use LilyPond backend
            conv = music21.converter.subConverters.ConverterLilypond()
            conv.write(score, fmt='lilypond', fp=output_path, subformats=['pdf'])
        else:
            # Use MuseScore backend (default)
            score.write('musicxml.pdf', fp=output_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to render PDF. Make sure MuseScore or LilyPond is "
            f"properly installed.\nError: {e}"
        )

    print(f"[midi_to_pdf] ✅ PDF saved: {output_path}")
    return output_path


def convert_midi_directory(
    input_dir: str,
    output_dir: str = './checkpoints/pdf_output',
) -> list:
    """Convert all MIDI files in a directory to PDF.

    Args:
        input_dir: Directory containing MIDI files.
        output_dir: Directory to save output PDFs.

    Returns:
        List of paths to generated PDF files.
    """
    midi_extensions = {'.mid', '.midi', '.MID', '.MIDI'}
    pdf_paths = []

    for filename in sorted(os.listdir(input_dir)):
        if os.path.splitext(filename)[1] in midi_extensions:
            midi_path = os.path.join(input_dir, filename)
            try:
                pdf_path = convert_midi_to_pdf(midi_path, output_dir)
                pdf_paths.append(pdf_path)
            except Exception as e:
                print(f"[midi_to_pdf] ⚠️  Skipping {filename}: {e}")

    print(f"\n[midi_to_pdf] Converted {len(pdf_paths)} files.")
    return pdf_paths


def main():
    """CLI entry point for MIDI to PDF conversion."""
    parser = argparse.ArgumentParser(
        description='Convert MIDI files to PDF sheet music using music21.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single MIDI file
  python -m mt3.midi_to_pdf song.mid

  # Specify output directory and title
  python -m mt3.midi_to_pdf song.mid -o ./output --title "My Song"

  # Convert all MIDI files in a directory
  python -m mt3.midi_to_pdf --batch ./midi_folder -o ./output

Requirements:
  - music21 (pip install music21)
  - MuseScore (https://musescore.org) or LilyPond (https://lilypond.org)
        """,
    )

    parser.add_argument(
        'midi_path',
        nargs='?',
        help='Path to the input MIDI file.',
    )
    parser.add_argument(
        '-o', '--output_dir',
        default='./checkpoints/pdf_output',
        help='Output directory for PDF files (default: ./checkpoints/pdf_output)',
    )
    parser.add_argument(
        '--title',
        default=None,
        help='Title to display on the sheet music.',
    )
    parser.add_argument(
        '--output_filename',
        default=None,
        help='Custom output filename (default: same as input with .pdf).',
    )
    parser.add_argument(
        '--batch',
        default=None,
        help='Directory of MIDI files to batch convert.',
    )

    args = parser.parse_args()

    if args.batch:
        if not os.path.isdir(args.batch):
            print(f"Error: Directory not found: {args.batch}")
            sys.exit(1)
        convert_midi_directory(args.batch, args.output_dir)
    elif args.midi_path:
        try:
            convert_midi_to_pdf(
                args.midi_path,
                output_dir=args.output_dir,
                title=args.title,
                output_filename=args.output_filename,
            )
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
