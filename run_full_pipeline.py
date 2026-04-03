"""
EvoCRM Full Pipeline Orchestrator — Olist Dataset
====================================================

Runs the ENTIRE pipeline from raw Olist CSVs to paper-ready results
in a single command. Each step validates the previous step's output
before proceeding.

Pipeline Steps:
    Step 0: Verify environment & dependencies
    Step 1: Run Olist data adapter
    Step 2: Validate adapter output
    Step 3: Train baselines (XGBoost, LightGBM, RF, Linear)
    Step 4: Train EvoCRM (3-phase: towers → hub → finetune)
    Step 5: Evaluate EvoCRM vs baselines
    Step 6: Run ablation studies
    Step 7: Generate paper artifacts (LaTeX tables + figures)

Usage:
    # Full pipeline
    python run_full_pipeline.py --olist_dir ./olist_raw/

    # Resume from a specific step (skips completed steps)
    python run_full_pipeline.py --olist_dir ./olist_raw/ --start_step 4

    # Quick mode (fewer epochs, for testing)
    python run_full_pipeline.py --olist_dir ./olist_raw/ --quick

Author: EvoCRM Team
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

class PipelineConfig:
    """Central configuration for the entire pipeline."""

    def __init__(self, args):
        # Directories
        self.olist_raw_dir = args.olist_dir
        self.work_dir = Path(args.work_dir)
        self.data_dir = self.work_dir / "data" / "olist"
        self.results_dir = self.work_dir / "results"
        self.baselines_dir = self.results_dir / "baselines"
        self.checkpoint_dir = self.work_dir / "checkpoints"
        self.eval_dir = self.results_dir / "evaluation"
        self.ablations_dir = self.results_dir / "ablations"
        self.paper_dir = self.results_dir / "paper_artifacts"

        # Training config
        self.quick = args.quick
        if self.quick:
            self.tower_epochs = 5
            self.hub_epochs = 3
            self.finetune_epochs = 3
            self.ablation_epochs = 3
            self.batch_size = 512
        else:
            self.tower_epochs = args.tower_epochs
            self.hub_epochs = args.hub_epochs
            self.finetune_epochs = args.finetune_epochs
            self.ablation_epochs = args.ablation_epochs
            self.batch_size = args.batch_size

        self.lr = args.lr
        self.lora_rank = args.lora_rank
        self.seed = args.seed
        self.start_step = args.start_step

    def create_dirs(self):
        """Create all output directories."""
        for d in [self.data_dir, self.results_dir, self.baselines_dir,
                  self.checkpoint_dir, self.eval_dir, self.ablations_dir,
                  self.paper_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP RUNNER
# ============================================================

class PipelineRunner:
    """Orchestrates the full EvoCRM pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.script_dir = Path(__file__).parent
        self.step_log = {}
        self.start_time = time.time()

    def run(self):
        """Execute all pipeline steps."""
        self.config.create_dirs()

        self._print_banner()

        steps = [
            (0, "Verify Environment", self.step0_verify_env),
            (1, "Olist Data Adapter", self.step1_adapter),
            (2, "Validate Adapter Output", self.step2_validate),
            (3, "Train Baselines", self.step3_baselines),
            (4, "Train EvoCRM (3-Phase)", self.step4_train_evocrm),
            (5, "Evaluate EvoCRM vs Baselines", self.step5_evaluate),
            (6, "Run Ablation Studies", self.step6_ablations),
            (7, "Generate Paper Artifacts", self.step7_paper_artifacts),
        ]

        for step_num, step_name, step_fn in steps:
            if step_num < self.config.start_step:
                print(f"\n  ⏭  Step {step_num}: {step_name} — SKIPPED (--start_step)")
                continue

            print(f"\n{'═' * 60}")
            print(f"  STEP {step_num}: {step_name}")
            print(f"{'═' * 60}")

            t0 = time.time()
            try:
                success = step_fn()
                elapsed = time.time() - t0
                self.step_log[step_num] = {
                    "name": step_name,
                    "status": "SUCCESS" if success else "FAILED",
                    "time_sec": round(elapsed, 1),
                }

                if success:
                    print(f"\n  ✓ Step {step_num} completed in {elapsed:.0f}s")
                else:
                    print(f"\n  ✗ Step {step_num} FAILED after {elapsed:.0f}s")
                    self._print_failure_advice(step_num)
                    if step_num <= 2:  # Critical steps — can't continue
                        print("  ABORTING: Cannot proceed without data.")
                        break

            except Exception as e:
                elapsed = time.time() - t0
                self.step_log[step_num] = {
                    "name": step_name,
                    "status": f"ERROR: {str(e)[:200]}",
                    "time_sec": round(elapsed, 1),
                }
                print(f"\n  ✗ Step {step_num} ERROR: {e}")
                import traceback
                traceback.print_exc()
                if step_num <= 2:
                    break

        self._print_final_summary()

    # ---- Step 0: Environment ----
    def step0_verify_env(self) -> bool:
        """Verify all dependencies are installed."""
        checks = {
            "numpy": "import numpy; print(numpy.__version__)",
            "pandas": "import pandas; print(pandas.__version__)",
            "sklearn": "import sklearn; print(sklearn.__version__)",
            "torch": "import torch; print(f'{torch.__version__} CUDA={torch.cuda.is_available()}')",
        }

        optional = {
            "xgboost": "import xgboost; print(xgboost.__version__)",
            "lightgbm": "import lightgbm; print(lightgbm.__version__)",
        }

        all_ok = True
        for name, cmd in checks.items():
            try:
                result = subprocess.run(
                    [sys.executable, "-c", cmd],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    print(f"  ✓ {name}: {result.stdout.strip()}")
                else:
                    print(f"  ✗ {name}: MISSING — pip install {name}")
                    all_ok = False
            except Exception as e:
                print(f"  ✗ {name}: ERROR — {e}")
                all_ok = False

        for name, cmd in optional.items():
            try:
                result = subprocess.run(
                    [sys.executable, "-c", cmd],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    print(f"  ✓ {name}: {result.stdout.strip()} (optional)")
                else:
                    print(f"  ⚠ {name}: not installed (optional, baselines will skip it)")
            except Exception:
                print(f"  ⚠ {name}: not available (optional)")

        # Check GPU
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 "import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('CPU only')"],
                capture_output=True, text=True, timeout=30,
            )
            print(f"  GPU: {result.stdout.strip()}")
        except Exception:
            print(f"  GPU: Unable to detect")

        # Check Olist data exists
        olist_path = Path(self.config.olist_raw_dir)
        required_files = [
            "olist_customers_dataset.csv",
            "olist_orders_dataset.csv",
            "olist_order_items_dataset.csv",
        ]
        for f in required_files:
            if (olist_path / f).exists():
                print(f"  ✓ {f}")
            else:
                print(f"  ✗ {f} — NOT FOUND in {olist_path}")
                all_ok = False

        if not all_ok:
            print("\n  FIX: Download Olist data first:")
            print("    kaggle datasets download -d olistbr/brazilian-ecommerce")
            print(f"    unzip brazilian-ecommerce.zip -d {self.config.olist_raw_dir}")

        return all_ok

    # ---- Step 1: Adapter ----
    def step1_adapter(self) -> bool:
        """Run Olist data adapter."""
        cmd = [
            sys.executable, str(self.script_dir / "olist_adapter.py"),
            "--data_dir", str(self.config.olist_raw_dir),
            "--output_dir", str(self.config.data_dir),
            "--seed", str(self.config.seed),
        ]
        return self._run_cmd(cmd, "olist_adapter")

    # ---- Step 2: Validate ----
    def step2_validate(self) -> bool:
        """Validate adapter output."""
        cmd = [
            sys.executable, str(self.script_dir / "validate_adapter_output.py"),
            "--data_dir", str(self.config.data_dir),
        ]
        return self._run_cmd(cmd, "validate")

    # ---- Step 3: Baselines ----
    def step3_baselines(self) -> bool:
        """Train all baselines."""
        cmd = [
            sys.executable, str(self.script_dir / "train_baselines.py"),
            "--data_dir", str(self.config.data_dir),
            "--output_dir", str(self.config.baselines_dir),
            "--seed", str(self.config.seed),
        ]
        return self._run_cmd(cmd, "baselines")

    # ---- Step 4: Train EvoCRM ----
    def step4_train_evocrm(self) -> bool:
        """Train EvoCRM in 3 phases."""
        epochs_str = (
            f"{self.config.tower_epochs},{self.config.hub_epochs},"
            f"{self.config.finetune_epochs}"
        )
        cmd = [
            sys.executable, str(self.script_dir / "train_evocrm.py"),
            "--data_dir", str(self.config.data_dir),
            "--phase", "all",
            "--epochs", epochs_str,
            "--batch_size", str(self.config.batch_size),
            "--lr", str(self.config.lr),
            "--use_lora",
            "--lora_rank", str(self.config.lora_rank),
            "--checkpoint_dir", str(self.config.checkpoint_dir),
            "--seed", str(self.config.seed),
        ]
        return self._run_cmd(cmd, "train_evocrm")

    # ---- Step 5: Evaluate ----
    def step5_evaluate(self) -> bool:
        """Evaluate EvoCRM vs baselines."""
        ckpt = self.config.checkpoint_dir / "evocrm_final.pt"
        baselines_file = self.config.baselines_dir / "all_baseline_results.json"

        if not ckpt.exists():
            print(f"  ERROR: Checkpoint not found: {ckpt}")
            return False

        cmd = [
            sys.executable, str(self.script_dir / "evaluate_evocrm.py"),
            "--data_dir", str(self.config.data_dir),
            "--checkpoint", str(ckpt),
            "--output_dir", str(self.config.eval_dir),
            "--batch_size", str(self.config.batch_size),
        ]
        if baselines_file.exists():
            cmd.extend(["--baselines", str(baselines_file)])

        return self._run_cmd(cmd, "evaluate")

    # ---- Step 6: Ablations ----
    def step6_ablations(self) -> bool:
        """Run ablation studies."""
        cmd = [
            sys.executable, str(self.script_dir / "run_ablations.py"),
            "--data_dir", str(self.config.data_dir),
            "--output_dir", str(self.config.ablations_dir),
            "--epochs", str(self.config.ablation_epochs),
            "--batch_size", str(self.config.batch_size),
            "--seed", str(self.config.seed),
        ]
        return self._run_cmd(cmd, "ablations")

    # ---- Step 7: Paper Artifacts ----
    def step7_paper_artifacts(self) -> bool:
        """Generate paper-ready tables and figures."""
        script = self.script_dir / "generate_paper_artifacts.py"
        if script.exists():
            cmd = [
                sys.executable, str(script),
                "--results_dir", str(self.config.results_dir),
                "--output_dir", str(self.config.paper_dir),
            ]
            return self._run_cmd(cmd, "paper_artifacts")
        else:
            # Inline generation if script not found
            return self._generate_paper_artifacts_inline()

    def _generate_paper_artifacts_inline(self) -> bool:
        """Generate paper artifacts without a separate script."""
        paper_dir = self.config.paper_dir
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Load all results
        eval_path = self.config.eval_dir / "evaluation_results.json"
        baselines_path = self.config.baselines_dir / "all_baseline_results.json"
        ablations_path = self.config.ablations_dir / "all_ablation_results.json"

        artifacts = {}

        if eval_path.exists():
            with open(eval_path) as f:
                eval_results = json.load(f)
            artifacts["evaluation"] = eval_results
            print(f"  ✓ Loaded evaluation results")

        if baselines_path.exists():
            with open(baselines_path) as f:
                baselines = json.load(f)
            artifacts["baselines"] = baselines
            print(f"  ✓ Loaded baseline results")

        if ablations_path.exists():
            with open(ablations_path) as f:
                ablations = json.load(f)
            artifacts["ablations"] = ablations
            print(f"  ✓ Loaded ablation results")

        # Save consolidated results
        with open(paper_dir / "all_results_consolidated.json", "w") as f:
            json.dump(artifacts, f, indent=2, default=str)

        # Generate summary text
        summary_lines = [
            "=" * 60,
            "PAPER-READY RESULTS SUMMARY",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            f"Dataset: Olist Brazilian E-Commerce",
            "",
        ]

        if "evaluation" in artifacts:
            summary_lines.append("MAIN RESULTS (Table 1):")
            summary_lines.append("-" * 40)
            evocrm = artifacts["evaluation"].get("evocrm", {})
            for task, metrics in evocrm.items():
                metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                         if isinstance(v, (int, float)))
                summary_lines.append(f"  {task}: {metric_str}")

            comparison = artifacts["evaluation"].get("comparison", {})
            if comparison:
                summary_lines.append("")
                summary_lines.append("DELTA vs BEST BASELINE:")
                summary_lines.append("-" * 40)
                for task, models in comparison.items():
                    for model, comp in models.items():
                        if comp.get("evocrm_wins"):
                            summary_lines.append(
                                f"  {task} vs {model}: "
                                f"Δ={comp['absolute_diff']:+.4f} "
                                f"({comp['relative_pct']:+.1f}%) ✓"
                            )

        summary_text = "\n".join(summary_lines)
        with open(paper_dir / "results_summary.txt", "w") as f:
            f.write(summary_text)

        print(f"\n{summary_text}")
        print(f"\n  Artifacts saved to: {paper_dir}")
        return True

    # ---- Helpers ----
    def _run_cmd(self, cmd: list, label: str) -> bool:
        """Run a command and stream output."""
        print(f"  Running: {' '.join(cmd[-6:])}")  # Show last 6 args
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                print(f"  {line}", end="")
            process.wait()
            return process.returncode == 0
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    def _print_banner(self):
        print("╔" + "═" * 58 + "╗")
        print("║         EvoCRM FULL PIPELINE — Olist Dataset            ║")
        print("║         From Raw Data → Paper-Ready Results             ║")
        print("╠" + "═" * 58 + "╣")
        mode = "QUICK (reduced epochs)" if self.config.quick else "FULL (production epochs)"
        print(f"║  Mode: {mode:<50}║")
        print(f"║  Olist data: {str(self.config.olist_raw_dir):<46}║")
        print(f"║  Work dir:   {str(self.config.work_dir):<46}║")
        print("╚" + "═" * 58 + "╝")

    def _print_failure_advice(self, step_num):
        advice = {
            0: "Install missing packages with pip. Ensure GPU drivers are working.",
            1: "Check Olist CSV file paths. Ensure all 7 required files are present.",
            2: "Adapter output is corrupted. Re-run Step 1.",
            3: "Baselines failed. Check scikit-learn version. Non-critical — can continue.",
            4: "EvoCRM training failed. Check GPU memory. Try --batch_size 128.",
            5: "Evaluation failed. Ensure checkpoint exists at checkpoints/evocrm_final.pt.",
            6: "Ablations failed. Non-critical for paper submission if main results exist.",
            7: "Paper artifact generation failed. Results still available in results/ dir.",
        }
        print(f"  ADVICE: {advice.get(step_num, 'Check logs above.')}")

    def _print_final_summary(self):
        total_time = time.time() - self.start_time
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║              PIPELINE EXECUTION SUMMARY                 ║")
        print("╠" + "═" * 58 + "╣")

        for step_num, info in sorted(self.step_log.items()):
            status = info["status"]
            icon = "✓" if status == "SUCCESS" else "✗"
            time_str = f"{info['time_sec']:.0f}s"
            print(f"║  {icon} Step {step_num}: {info['name']:<38} {time_str:>6} ║")

        print("╠" + "═" * 58 + "╣")
        print(f"║  Total time: {total_time:.0f}s ({total_time/60:.1f} min)"
              f"{'':>30}║")
        print("╚" + "═" * 58 + "╝")

        # Point to key output files
        print("\n  KEY OUTPUT FILES:")
        key_files = [
            (self.config.baselines_dir / "all_baseline_results.json", "Baseline numbers"),
            (self.config.checkpoint_dir / "evocrm_final.pt", "Trained model"),
            (self.config.eval_dir / "evaluation_results.json", "EvoCRM vs baselines (TABLE 1)"),
            (self.config.eval_dir / "table1_main_results.tex", "LaTeX table for paper"),
            (self.config.ablations_dir / "all_ablation_results.json", "Ablation results"),
            (self.config.paper_dir / "results_summary.txt", "Paper results summary"),
        ]
        for path, desc in key_files:
            exists = "✓" if path.exists() else "✗"
            print(f"  {exists} {path} — {desc}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the complete EvoCRM pipeline on Olist data"
    )
    parser.add_argument("--olist_dir", type=str, default="./olist_raw/",
                        help="Path to raw Olist CSV files")
    parser.add_argument("--work_dir", type=str, default="./evocrm_workspace/",
                        help="Working directory for all outputs")
    parser.add_argument("--start_step", type=int, default=0,
                        help="Resume from this step number (0-7)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer epochs for testing")

    # Training hyperparameters
    parser.add_argument("--tower_epochs", type=int, default=50)
    parser.add_argument("--hub_epochs", type=int, default=30)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--ablation_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = PipelineConfig(args)
    runner = PipelineRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
