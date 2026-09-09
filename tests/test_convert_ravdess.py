import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from emotion_calibration.convert_ravdess import (
    RavdessFilenameError,
    convert,
    parse_ravdess_filename,
)


class ParseRavdessFilenameTests(unittest.TestCase):
    def test_parses_all_fields(self) -> None:
        row = parse_ravdess_filename("Actor_12/03-01-05-02-02-01-12.wav")

        self.assertEqual(row["emotion"], "angry")
        self.assertEqual(row["vocal_channel"], "speech")
        self.assertEqual(row["intensity"], "strong")
        self.assertEqual(row["actor"], 12)
        self.assertEqual(row["actor_gender"], "female")

    def test_rejects_invalid_filename(self) -> None:
        with self.assertRaises(RavdessFilenameError):
            parse_ravdess_filename("audio.wav")

    def test_converts_zip_and_filters_emotions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "ravdess.zip"
            output = root / "labels.csv"
            with ZipFile(archive, "w") as zip_file:
                zip_file.writestr("Actor_01/03-01-03-01-01-01-01.wav", b"")
                zip_file.writestr("Actor_02/03-01-04-01-01-01-02.wav", b"")

            count = convert(archive, output, {"happy"})

            self.assertEqual(count, 1)
            contents = output.read_text(encoding="utf-8")
            self.assertIn("happy", contents)
            self.assertNotIn("sad", contents)


if __name__ == "__main__":
    unittest.main()
