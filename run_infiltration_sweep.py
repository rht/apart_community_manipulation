#!/usr/bin/env python3
"""
Meta-script to run infiltration experiments across a range of infiltrator counts
and plot the results.

Automatically resumes from existing results if sweep_results.json exists in the
output directory, skipping experiments that have already been run with the same
parameters.

Usage:
    python examples/run_infiltration_sweep.py

    # Custom range
    python examples/run_infiltration_sweep.py --min-infiltrators 1 --max-infiltrators 10

    # Skip plotting (just collect data)
    python examples/run_infiltration_sweep.py --no-plot

    # Force fresh run, ignoring existing results
    python examples/run_infiltration_sweep.py --force-rerun
"""

import argparse
import asyncio
import json
import os
from datetime import datetime

from dotenv import load_dotenv

from cm_library import (
    CommunityInfiltrationSimulation,
    InfiltrationConfig,
)

load_dotenv()

# Model configuration - edit these to change which models are used
STRONG_MODEL = "openai/gpt-4o"  # Infiltrator agents
WEAK_MODEL = "google/gemini-2.0-flash-001"  # Population agents
# WEAK_MODEL = "openai/gpt-4o-mini"


def sanitize_model_name(model: str) -> str:
    """Convert model name to filesystem-safe string."""
    return model.replace("/", "_").replace(":", "_")


def load_existing_results(output_dir: str, model_prefix: str) -> dict | None:
    """Load existing results from output directory if available."""
    results_path = f"{output_dir}/{model_prefix}_sweep_results.json"
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing results: {e}")
    return None


async def run_sweep(
    min_infiltrators: int = 1,
    max_infiltrators: int = 10,
    step: int = 1,
    num_population: int = 20,
    max_timesteps: int = 50,
    belief_check_interval: int = 5,
    num_trials: int = 1,
    output_dir: str = "./data/sweep_results",
    force_rerun: bool = False,
    strong_model: str = STRONG_MODEL,
    weak_model: str = WEAK_MODEL,
    mode: str = "targeted",
) -> dict:
    """
    Run infiltration experiments across a range of infiltrator counts.

    Args:
        min_infiltrators: Minimum number of infiltrators to test
        max_infiltrators: Maximum number of infiltrators to test
        step: Step size for infiltrator count
        num_population: Number of population agents (constant across experiments)
        max_timesteps: Maximum timesteps per experiment
        belief_check_interval: How often to check beliefs
        num_trials: Number of trials per infiltrator count (for averaging)
        output_dir: Directory to save results
        force_rerun: If True, ignore existing results and run all experiments fresh
        strong_model: Model name for infiltrator agents
        weak_model: Model name for population agents
        mode: Infiltrator action mode (targeted, broadcast_only, or llm_action_only)

    Returns:
        Dictionary with sweep results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create sanitized model name prefix for filenames
    strong_model_safe = sanitize_model_name(strong_model)
    weak_model_safe = sanitize_model_name(weak_model)
    mode_suffix = f"_{mode}" if mode != "targeted" else ""
    model_prefix = f"{strong_model_safe}_vs_{weak_model_safe}{mode_suffix}"

    infiltrator_counts = list(range(min_infiltrators, max_infiltrators + 1, step))

    # Try to load existing results
    existing_results = (
        None if force_rerun else load_existing_results(output_dir, model_prefix)
    )
    existing_experiments = {}
    if existing_results:
        for exp in existing_results.get("experiments", []):
            existing_experiments[exp["num_infiltrators"]] = exp

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_population": num_population,
            "max_timesteps": max_timesteps,
            "belief_check_interval": belief_check_interval,
            "num_trials": num_trials,
            "infiltrator_counts": infiltrator_counts,
            "strong_model": strong_model,
            "weak_model": weak_model,
            "mode": mode,
        },
        "experiments": [],
    }

    # Determine which experiments need to be run
    counts_to_run = []
    counts_skipped = []
    for num_infiltrators in infiltrator_counts:
        if num_infiltrators in existing_experiments:
            existing_exp = existing_experiments[num_infiltrators]
            # Check if the existing experiment has the same parameters
            if (
                existing_results
                and existing_results["metadata"].get("num_population") == num_population
                and existing_results["metadata"].get("max_timesteps") == max_timesteps
                and existing_results["metadata"].get("num_trials") == num_trials
            ):
                counts_skipped.append(num_infiltrators)
                results["experiments"].append(existing_exp)
            else:
                counts_to_run.append(num_infiltrators)
        else:
            counts_to_run.append(num_infiltrators)

    print("\n" + "=" * 70)
    print("INFILTRATION SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"Infiltrator range: {min_infiltrators} to {max_infiltrators} (step={step})")
    print(f"Population size: {num_population}")
    print(f"Max timesteps: {max_timesteps}")
    print(f"Trials per config: {num_trials}")
    if counts_skipped:
        print(f"Reusing existing results for: {counts_skipped}")
    if counts_to_run:
        print(f"Running experiments for: {counts_to_run}")
    else:
        print("All experiments already completed!")
    print("=" * 70 + "\n")

    for num_infiltrators in counts_to_run:
        trial_results = []

        for trial in range(num_trials):
            print(f"\n[{num_infiltrators} infiltrators] Trial {trial + 1}/{num_trials}")
            print("-" * 50)

            config = InfiltrationConfig(
                enable_targeted_commenting=(mode == "targeted"),
                broadcast_only=(mode == "broadcast_only"),
                llm_action_only=(mode == "llm_action_only"),
                use_llm_belief_analysis=True,
                num_infiltrators=num_infiltrators,
                num_population=num_population,
                max_timesteps=max_timesteps,
                belief_check_interval=belief_check_interval,
                strong_model=strong_model,
                weak_model=weak_model,
                db_path=f"{output_dir}/{model_prefix}_infiltration_{num_infiltrators}_trial{trial}.db",
                checkpoint_path=f"{output_dir}/{model_prefix}_checkpoint_{num_infiltrators}_trial{trial}.json",
            )

            sim = CommunityInfiltrationSimulation(config)
            result = await sim.run()

            trial_results.append(
                {
                    "trial": trial,
                    "timesteps_to_full_conviction": result.timesteps_to_full_conviction,
                    "final_conviction_rate": result.final_conviction_rate,
                    "conviction_history": result.conviction_history,
                    # Comment-based metrics
                    "timesteps_to_full_conviction_comments": result.timesteps_to_full_conviction_comments,
                    "comment_final_conviction_rate": result.comment_final_conviction_rate,
                    "comment_conviction_history": result.comment_conviction_history,
                }
            )

        # Aggregate trial results (interview-based)
        successful_trials = [
            t for t in trial_results if t["timesteps_to_full_conviction"] is not None
        ]

        if successful_trials:
            avg_timesteps = sum(
                t["timesteps_to_full_conviction"] for t in successful_trials
            ) / len(successful_trials)
            min_timesteps = min(
                t["timesteps_to_full_conviction"] for t in successful_trials
            )
            max_timesteps_result = max(
                t["timesteps_to_full_conviction"] for t in successful_trials
            )
        else:
            avg_timesteps = None
            min_timesteps = None
            max_timesteps_result = None

        avg_final_rate = sum(t["final_conviction_rate"] for t in trial_results) / len(
            trial_results
        )

        # Aggregate comment-based metrics
        successful_trials_comments = [
            t
            for t in trial_results
            if t["timesteps_to_full_conviction_comments"] is not None
        ]

        if successful_trials_comments:
            avg_timesteps_comments = sum(
                t["timesteps_to_full_conviction_comments"]
                for t in successful_trials_comments
            ) / len(successful_trials_comments)
        else:
            avg_timesteps_comments = None

        avg_comment_final_rate = sum(
            t["comment_final_conviction_rate"] for t in trial_results
        ) / len(trial_results)

        experiment_result = {
            "num_infiltrators": num_infiltrators,
            "num_trials": num_trials,
            # Interview-based metrics
            "success_rate": len(successful_trials) / num_trials,
            "avg_timesteps_to_conviction": avg_timesteps,
            "min_timesteps": min_timesteps,
            "max_timesteps": max_timesteps_result,
            "avg_final_conviction_rate": avg_final_rate,
            # Comment-based metrics
            "success_rate_comments": len(successful_trials_comments) / num_trials,
            "avg_timesteps_to_conviction_comments": avg_timesteps_comments,
            "avg_comment_final_conviction_rate": avg_comment_final_rate,
            "trials": trial_results,
        }
        results["experiments"].append(experiment_result)

        print(
            f"\n>>> {num_infiltrators} infiltrators: "
            f"interview={avg_final_rate:.1%}, "
            f"comments={avg_comment_final_rate:.1%}"
        )

    # Sort experiments by infiltrator count for consistent ordering
    results["experiments"].sort(key=lambda x: x["num_infiltrators"])

    # Save results
    results_path = f"{output_dir}/{model_prefix}_sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def plot_results(
    results: dict,
    output_dir: str = "./data/sweep_results",
    mode: str = "targeted",
):
    """Generate plots from sweep results comparing interview and comment-based analysis."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nMatplotlib not installed. Install with: pip install matplotlib")
        print("Skipping plot generation.")
        return

    experiments = results["experiments"]

    # Extract data for plotting
    infiltrator_counts = [e["num_infiltrators"] for e in experiments]

    # Interview-based metrics
    final_rates = [e["avg_final_conviction_rate"] for e in experiments]

    # Comment-based metrics (with fallback for old data)
    comment_final_rates = [
        e.get("avg_comment_final_conviction_rate", 0) for e in experiments
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Final conviction rate comparison (Interview vs Comments)
    ax1 = axes[0]
    x = np.arange(len(infiltrator_counts))
    width = 0.35
    bars1 = ax1.bar(
        x - width / 2,
        [r * 100 for r in final_rates],
        width,
        label="Interview",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x + width / 2,
        [r * 100 for r in comment_final_rates],
        width,
        label="Comments",
        color="coral",
        alpha=0.8,
    )
    ax1.set_xlabel("Number of Infiltrators", fontsize=12)
    ax1.set_ylabel("Final Conviction Rate (%)", fontsize=12)
    ax1.set_title("Final Conviction Rate: Interview vs Comments", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(infiltrator_counts)
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Interview conviction curves over time
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
    for i, exp in enumerate(experiments):
        if exp["trials"]:
            history = exp["trials"][0]["conviction_history"]
            if history:
                timesteps = [h[0] for h in history]
                convinced = [
                    h[1] / results["metadata"]["num_population"] * 100 for h in history
                ]
                ax2.plot(
                    timesteps,
                    convinced,
                    "-o",
                    color=colors[i],
                    label=f"{exp['num_infiltrators']} inf",
                    markersize=4,
                )
    ax2.set_xlabel("Timestep", fontsize=12)
    ax2.set_ylabel("Conviction Rate (%)", fontsize=12)
    ax2.set_title("Interview-Based Conviction Over Time", fontsize=14)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Comment conviction curves over time
    ax3 = axes[2]
    for i, exp in enumerate(experiments):
        if exp["trials"]:
            history = exp["trials"][0].get("comment_conviction_history", [])
            if history:
                timesteps = [h[0] for h in history]
                convinced = [
                    h[1] / results["metadata"]["num_population"] * 100 for h in history
                ]
                ax3.plot(
                    timesteps,
                    convinced,
                    "-s",
                    color=colors[i],
                    label=f"{exp['num_infiltrators']} inf",
                    markersize=4,
                )
    ax3.set_xlabel("Timestep", fontsize=12)
    ax3.set_ylabel("Conviction Rate (%)", fontsize=12)
    ax3.set_title("Comment-Based Conviction Over Time", fontsize=14)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create model prefix for filenames
    strong_model = results["metadata"].get("strong_model", "unknown")
    weak_model = results["metadata"].get("weak_model", "unknown")
    mode_suffix = f"_{mode}" if mode != "targeted" else ""
    model_prefix = f"{sanitize_model_name(strong_model)}_vs_{sanitize_model_name(weak_model)}{mode_suffix}"

    # Save plot
    plot_path = f"{output_dir}/{model_prefix}_infiltration_sweep_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")


def print_summary_table(results: dict):
    """Print a summary table of results comparing interview and comment-based metrics."""
    print("\n" + "=" * 100)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 100)
    print(
        f"{'Infiltrators':<12} "
        f"{'Interview Rate':<16} "
        f"{'Comment Rate':<14} "
        f"{'Interview Success':<18} "
        f"{'Comment Success':<16}"
    )
    print("-" * 100)

    for exp in results["experiments"]:
        interview_rate = exp["avg_final_conviction_rate"] * 100
        comment_rate = exp.get("avg_comment_final_conviction_rate", 0) * 100
        interview_success = exp["success_rate"] * 100
        comment_success = exp.get("success_rate_comments", 0) * 100

        print(
            f"{exp['num_infiltrators']:<12} "
            f"{interview_rate:>6.1f}%{'':<8} "
            f"{comment_rate:>6.1f}%{'':<6} "
            f"{interview_success:>6.0f}%{'':<10} "
            f"{comment_success:>6.0f}%"
        )

    print("=" * 100)


async def main():
    parser = argparse.ArgumentParser(
        description="Run infiltration sweep across different infiltrator counts"
    )
    parser.add_argument(
        "--min-infiltrators", type=int, default=1, help="Minimum number of infiltrators"
    )
    parser.add_argument(
        "--max-infiltrators", type=int, default=5, help="Maximum number of infiltrators"
    )
    parser.add_argument(
        "--step", type=int, default=2, help="Step size for infiltrator count"
    )
    parser.add_argument(
        "--population", type=int, default=10, help="Number of population agents"
    )
    parser.add_argument(
        "--max-timesteps", type=int, default=20, help="Maximum timesteps per experiment"
    )
    parser.add_argument(
        "--check-interval", type=int, default=5, help="Belief check interval"
    )
    parser.add_argument(
        "--trials", type=int, default=1, help="Number of trials per infiltrator count"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/sweep_results",
        help="Output directory for results",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    parser.add_argument(
        "--plot-only",
        type=str,
        default=None,
        help="Only generate plots from existing results file",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing results and run all experiments fresh",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["targeted", "broadcast_only", "llm_action_only"],
        default="targeted",
        help="Infiltrator action mode: targeted (default), broadcast_only, or llm_action_only",
    )

    args = parser.parse_args()

    if args.plot_only:
        # Just load and plot existing results
        with open(args.plot_only, "r") as f:
            results = json.load(f)
        print_summary_table(results)
        plot_results(results, os.path.dirname(args.plot_only))
        return

    # Run sweep
    results = await run_sweep(
        min_infiltrators=args.min_infiltrators,
        max_infiltrators=args.max_infiltrators,
        step=args.step,
        num_population=args.population,
        max_timesteps=args.max_timesteps,
        belief_check_interval=args.check_interval,
        num_trials=args.trials,
        output_dir=args.output_dir,
        force_rerun=args.force_rerun,
        strong_model=STRONG_MODEL,
        weak_model=WEAK_MODEL,
        mode=args.mode,
    )

    # Print summary
    print_summary_table(results)

    # Generate plots
    if not args.no_plot:
        plot_results(results, args.output_dir, args.mode)


if __name__ == "__main__":
    asyncio.run(main())
