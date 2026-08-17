import argparse
import sys
import tempfile
import unittest
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from run_supplementary_experiments import (  # noqa: E402
    RunSpec,
    build_specs,
    merge_existing_runs,
    suite_methods,
)
from summarize_supplementary import aggregate  # noqa: E402
from compare_significance import bootstrap_ci, compare, sign_flip_pvalue  # noqa: E402


class SupplementaryLauncherTest(unittest.TestCase):
    def test_suite_sizes(self):
        self.assertEqual(len(list(suite_methods("core"))), 5)
        self.assertEqual(len(list(suite_methods("noise"))), 3)
        self.assertEqual(len(list(suite_methods("schedule"))), 5)
        self.assertEqual(len(list(suite_methods("sensitivity"))), 15)
        self.assertEqual(len(list(suite_methods("efficiency"))), 2)

    def test_manifest_merge_preserves_other_suites(self):
        old = RunSpec("core", "baseline", "chair", 0, "old", {}, ["old"])
        fresh = RunSpec("noise", "noise_fixed", "chair", 0, "new", {}, ["new"])
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "run_manifest.json"
            manifest.write_text(
                __import__("json").dumps({"runs": [__import__("dataclasses").asdict(old)]}),
                encoding="utf-8",
            )
            merged = merge_existing_runs(manifest, [fresh])
        self.assertEqual({item.suite for item in merged}, {"core", "noise"})

    def test_build_specs_cross_product(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                suite="noise",
                data_root=directory,
                output_root=directory,
                scenes=["chair", "sofa"],
                seeds=[0, 1, 2],
                python="python",
                trainer="simple_trainer_ours.py",
                exp_name="low",
                save_eval_images=False,
                render_trajectory=False,
                max_runs=None,
            )
            specs = build_specs(args)
        self.assertEqual(len(specs), 18)
        self.assertIn("--no-save-eval-images", specs[0].command)
        self.assertIn("--no-render-trajectory", specs[0].command)


class SupplementarySummaryTest(unittest.TestCase):
    def test_dataset_mean_std_is_across_seed_means(self):
        rows = []
        for seed, values in ((0, (10.0, 20.0)), (1, (12.0, 22.0))):
            for scene, psnr in zip(("a", "b"), values):
                rows.append(
                    {
                        "suite": "core",
                        "method": "full",
                        "scene": scene,
                        "seed": seed,
                        "psnr": psnr,
                        "num_GS": 100 + seed,
                    }
                )
        summaries = aggregate(rows)
        dataset = next(row for row in summaries if row["scene"] == "MEAN")
        self.assertEqual(dataset["n"], 2)
        self.assertAlmostEqual(dataset["psnr_mean"], 16.0)
        self.assertAlmostEqual(dataset["num_GS_mean"], 201.0)


class SignificanceTest(unittest.TestCase):
    def test_identical_values_have_unit_pvalue(self):
        self.assertEqual(sign_flip_pvalue([0.0, 0.0], 100, 1), 1.0)

    def test_bootstrap_is_deterministic(self):
        self.assertEqual(bootstrap_ci([1.0, 2.0, 3.0], 200, 7), bootstrap_ci([1.0, 2.0, 3.0], 200, 7))

    def test_pairs_match_scene_and_seed(self):
        rows = [
            {"suite": "core", "method": "baseline", "scene": "a", "seed": 0, "psnr": 10.0},
            {"suite": "core", "method": "full", "scene": "a", "seed": 0, "psnr": 11.0},
            {"suite": "core", "method": "full", "scene": "b", "seed": 0, "psnr": 99.0},
        ]
        result = compare(rows, "baseline", 100, 1)
        psnr = next(row for row in result if row["metric"] == "psnr")
        self.assertEqual(psnr["n_pairs"], 1)
        self.assertEqual(psnr["mean_delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
