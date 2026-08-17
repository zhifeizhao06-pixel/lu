import sys
import unittest
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from experiment_utils import curriculum_weight  # noqa: E402


class CurriculumWeightTest(unittest.TestCase):
    def test_boundaries(self):
        for schedule in ("step", "linear", "smoothstep", "exponential"):
            self.assertEqual(curriculum_weight(10, 10, 20, schedule), 0.0)
            self.assertEqual(curriculum_weight(20, 10, 20, schedule), 1.0)

    def test_fixed_is_always_active(self):
        self.assertEqual(curriculum_weight(-100, 10, 20, "fixed"), 1.0)
        self.assertEqual(curriculum_weight(100, 10, 20, "fixed"), 1.0)

    def test_schedules_are_monotone(self):
        for schedule in ("linear", "smoothstep", "exponential"):
            values = [curriculum_weight(step, 0, 100, schedule) for step in range(101)]
            self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))

    def test_step_location(self):
        self.assertEqual(curriculum_weight(49, 0, 100, "step"), 0.0)
        self.assertEqual(curriculum_weight(50, 0, 100, "step"), 1.0)
        self.assertEqual(curriculum_weight(79, 0, 100, "step", step_iteration=80), 0.0)
        self.assertEqual(curriculum_weight(80, 0, 100, "step", step_iteration=80), 1.0)

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            curriculum_weight(1, 5, 5, "linear")


if __name__ == "__main__":
    unittest.main()
