"""Convert RAVDESS filenames from a directory or ZIP archive into CSV labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path, PurePosixPath
import sys
from typing import Iterable
from zipfile import BadZipFile, ZipFile


MODALITIES = {
    "01": "full_av",
    "02": "video_only",
    "03": "audio_only",
}

VOCAL_CHANNELS = {
    "01": "speech",
    "02": "song",
}

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITIES = {
    "01": "normal",
    "02": "strong",
}

STATEMENTS = {
    "01": "kids_are_talking_by_the_door",
    "02": "dogs_are_sitting_by_the_door",
}

REPETITIONS = {
    "01": 1,
    "02": 2,
}

CSV_FIELDS = (
    "path",
    "filename",
    "modality",
    "vocal_channel",
    "emotion",
    "intensity",
    "statement",
    "repetition",
    "actor",
    "actor_gender",
)


class RavdessFilenameError(ValueError):
    """Raised when a filename does not follow the RAVDESS convention."""


def _decode(mapping: dict[str, object], code: str, field: str) -> object:
    try:
        return mapping[code]
    except KeyError as exc:
        raise RavdessFilenameError(f"unknown {field} code: {code}") from exc


def parse_ravdess_filename(path: str) -> dict[str, object]:
    """Parse one RAVDESS WAV filename into human-readable metadata."""
    filename = PurePosixPath(path.replace("\\", "/")).name
    if not filename.lower().endswith(".wav"):
        raise RavdessFilenameError("file must have the .wav extension")

    parts = filename[:-4].split("-")
    if len(parts) != 7 or any(len(part) != 2 or not part.isdigit() for part in parts):
        raise RavdessFilenameError(
            "expected seven two-digit codes, for example "
            "03-01-05-02-02-01-12.wav"
        )

    modality, channel, emotion, intensity, statement, repetition, actor_code = parts
    actor = int(actor_code)
    if not 1 <= actor <= 24:
        raise RavdessFilenameError(f"actor must be between 01 and 24: {actor_code}")

    return {
        "path": path,
        "filename": filename,
        "modality": _decode(MODALITIES, modality, "modality"),
        "vocal_channel": _decode(VOCAL_CHANNELS, channel, "vocal channel"),
        "emotion": _decode(EMOTIONS, emotion, "emotion"),
        "intensity": _decode(INTENSITIES, intensity, "intensity"),
        "statement": _decode(STATEMENTS, statement, "statement"),
        "repetition": _decode(REPETITIONS, repetition, "repetition"),
        "actor": actor,
        "actor_gender": "male" if actor % 2 else "female",
    }


def iter_wav_paths(source: Path) -> Iterable[str]:
    """Yield normalized WAV paths from a directory or a ZIP archive."""
    if source.is_dir():
        for file in sorted(source.rglob("*.wav")):
            yield file.relative_to(source).as_posix()
        return

    if source.is_file() and source.suffix.lower() == ".zip":
        try:
            with ZipFile(source) as archive:
                for name in sorted(archive.namelist()):
                    if not name.endswith("/") and name.lower().endswith(".wav"):
                        yield name
        except BadZipFile as exc:
            raise ValueError(f"invalid ZIP archive: {source}") from exc
        return

    raise ValueError("source must be a directory or a .zip archive")


def convert(source: Path, output: Path, selected_emotions: set[str] | None) -> int:
    """Read RAVDESS filenames, validate them and write a CSV file."""
    rows: list[dict[str, object]] = []
    errors: list[str] = []

    for wav_path in iter_wav_paths(source):
        try:
            row = parse_ravdess_filename(wav_path)
        except RavdessFilenameError as exc:
            errors.append(f"{wav_path}: {exc}")
            continue
        if selected_emotions is None or row["emotion"] in selected_emotions:
            rows.append(row)

    if errors:
        preview = "\n".join(errors[:5])
        suffix = "" if len(errors) <= 5 else f"\n... and {len(errors) - 5} more"
        raise ValueError(f"invalid RAVDESS filenames:\n{preview}{suffix}")
    if not rows:
        raise ValueError("no matching WAV files found")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a CSV label table from RAVDESS WAV filenames."
    )
    parser.add_argument("source", type=Path, help="RAVDESS directory or ZIP archive")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ravdess_labels.csv"),
        help="output CSV path (default: data/processed/ravdess_labels.csv)",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        choices=sorted(EMOTIONS.values()),
        help="optional list of emotions to keep",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        count = convert(
            source=args.source.expanduser(),
            output=args.output,
            selected_emotions=set(args.emotions) if args.emotions else None,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Created {args.output} with {count} rows")


if __name__ == "__main__":
    main()
