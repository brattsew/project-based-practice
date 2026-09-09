import unittest

from emotion_calibration.convert_ravdess import EMOTIONS, parse_filename


class ProjectTests(unittest.TestCase):
    def test_filename_is_decoded(self) -> None:
        label = parse_filename("Actor_12/03-01-05-02-02-01-12.wav")
        self.assertEqual(label["emotion"], "angry")
        self.assertEqual(label["intensity"], "strong")
        self.assertEqual(label["actor"], 12)

    def test_all_emotions_are_supported(self) -> None:
        self.assertEqual(len(EMOTIONS), 8)


if __name__ == "__main__":
    unittest.main()
