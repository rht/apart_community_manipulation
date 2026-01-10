"""
Community Manipulation Simulation

This simulation measures how long it takes for a small number of "infiltrator"
agents (using strong LLMs) to convince a larger community of "population" agents
(using weaker LLMs) of a particular idea/belief.

Similar to historical patterns like the Spanish Inquisition's spread in Latin America,
where a small group with strong conviction influenced a larger population.
"""

import asyncio
import json
import os
from datetime import datetime

from dotenv import load_dotenv
import cm_library

load_dotenv()


async def run_experiment_suite(
    infiltrator_counts: list = None,
    num_population: int = 20,
    max_timesteps: int = 50,
    belief_check_interval: int = 5,
    output_path: str = "./data/infiltration_results.json",
    max_concurrent_calls: int = 20,
    strong_model: str = "openai/gpt-4o",
    weak_model: str = "openai/gpt-4o-mini",
):
    """
    Run multiple experiments varying the number of infiltrators.
    """
    if infiltrator_counts is None:
        infiltrator_counts = [1, 2, 3, 5, 8, 10]

    all_results = []

    print("\n" + "=" * 70)
    print("COMMUNITY MANIPULATION EXPERIMENT SUITE")
    print("=" * 70)
    print(f"Testing infiltrator counts: {infiltrator_counts}")
    print(f"Population size: {num_population}")
    print(f"Max timesteps: {max_timesteps}")
    print(f"Max concurrent calls: {max_concurrent_calls}")
    print(f"Models: {strong_model} / {weak_model}")
    print("=" * 70 + "\n")

    suite_start = datetime.now()

    for num_infiltrators in infiltrator_counts:
        print(f"\n{'#' * 60}")
        print(f"# Experiment: {num_infiltrators} infiltrators")
        print(f"{'#' * 60}")

        config = cm_library.InfiltrationConfig(
            num_infiltrators=num_infiltrators,
            num_population=num_population,
            max_timesteps=max_timesteps,
            belief_check_interval=belief_check_interval,
            db_path=f"./data/infiltration_{num_infiltrators}_infiltrators.db",
            max_concurrent_calls=max_concurrent_calls,
            strong_model=strong_model,
            weak_model=weak_model,
        )

        sim = cm_library.CommunityInfiltrationSimulation(config)
        result = await sim.run()

        result_dict = {
            "num_infiltrators": result.num_infiltrators,
            "num_population": result.num_population,
            "timesteps_to_full_conviction": result.timesteps_to_full_conviction,
            "conviction_history": result.conviction_history,
            "final_conviction_rate": result.final_conviction_rate,
            "runtime_seconds": result.total_runtime_seconds,
        }
        all_results.append(result_dict)

        print(
            f"\nResult: {num_infiltrators} infiltrators -> "
            f"{'Full conviction at t=' + str(result.timesteps_to_full_conviction) if result.timesteps_to_full_conviction else 'Not achieved'}, "
            f"final rate: {result.final_conviction_rate:.1%}, "
            f"runtime: {result.total_runtime_seconds:.1f}s"
        )

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    suite_duration = (datetime.now() - suite_start).total_seconds()

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(
        f"{'Infiltrators':<15} {'Time to Full':<20} {'Final Rate':<15} {'Runtime':<10}"
    )
    print("-" * 60)
    for r in all_results:
        time_str = (
            str(r["timesteps_to_full_conviction"])
            if r["timesteps_to_full_conviction"]
            else "N/A"
        )
        print(
            f"{r['num_infiltrators']:<15} {time_str:<20} {r['final_conviction_rate']:.1%}{'':8} {r['runtime_seconds']:.1f}s"
        )
    print("=" * 70)
    print(f"Total suite runtime: {suite_duration:.1f}s")
    print(f"\nResults saved to: {output_path}")

    return all_results


async def main():
    """Run a single experiment or the full suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Community manipulation Simulation")
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single experiment instead of the full suite",
    )
    parser.add_argument(
        "--infiltrators",
        type=int,
        default=3,
        help="Number of infiltrators (for single experiment)",
    )
    parser.add_argument(
        "--population", type=int, default=20, help="Number of population agents"
    )
    parser.add_argument(
        "--max-timesteps", type=int, default=10, help="Maximum timesteps"
    )
    parser.add_argument(
        "--check-interval", type=int, default=5, help="Belief check interval"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum concurrent API calls (default: 20)",
    )
    parser.add_argument(
        "--strong-model",
        type=str,
        default="openai/gpt-4o",
        help="Model for infiltrators (default: openai/gpt-4o)",
    )
    parser.add_argument(
        "--weak-model",
        type=str,
        default="google/gemini-2.0-flash-001",
        help="Model for population (default: google/gemini-2.0-flash-001)",
    )

    args = parser.parse_args()

    if args.single:
        # Single experiment
        config = cm_library.InfiltrationConfig(
            num_infiltrators=args.infiltrators,
            num_population=args.population,
            max_timesteps=args.max_timesteps,
            belief_check_interval=args.check_interval,
            resume=not args.fresh,
            max_concurrent_calls=args.max_concurrent,
            strong_model=args.strong_model,
            weak_model=args.weak_model,
        )
        sim = cm_library.CommunityInfiltrationSimulation(config)
        result = await sim.run()

        print("\n" + "=" * 50)
        print("SINGLE EXPERIMENT RESULT")
        print("=" * 50)
        print(f"Infiltrators: {result.num_infiltrators}")
        print(f"Population: {result.num_population}")
        print(
            f"Time to full conviction: {result.timesteps_to_full_conviction or 'N/A'}"
        )
        print(f"Final conviction rate: {result.final_conviction_rate:.1%}")
        print(f"Conviction history: {result.conviction_history}")
        print(f"Total runtime: {result.total_runtime_seconds:.1f}s")
    else:
        # Full experiment suite
        await run_experiment_suite(
            infiltrator_counts=[1, 2, 3, 5, 8],
            num_population=args.population,
            max_timesteps=args.max_timesteps,
            belief_check_interval=args.check_interval,
            max_concurrent_calls=args.max_concurrent,
            strong_model=args.strong_model,
            weak_model=args.weak_model,
        )


if __name__ == "__main__":
    asyncio.run(main())
