#!/usr/bin/env python3
"""
Visualize belief propagation through a community network like a virus contagion.

Creates animated or multi-panel visualization showing how beliefs spread from
infiltrators to population agents over time.

Usage:
    python plot_belief_contagion.py --db path/to/db.db --results path/to/results.json
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def load_database_data(db_path: str) -> dict:
    """Load users, follow relationships, and belief data from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all users
    cursor.execute("SELECT user_id, agent_id, user_name, name FROM user")
    users = {row[0]: {"agent_id": row[1], "user_name": row[2], "name": row[3]} for row in cursor.fetchall()}

    # Get follow relationships (the actual graph structure)
    cursor.execute("""
        SELECT follower_id, followee_id
        FROM follow
    """)
    follows = cursor.fetchall()

    # Get interview responses over time
    cursor.execute("""
        SELECT user_id, created_at, info
        FROM trace
        WHERE action = 'interview'
        ORDER BY created_at
    """)
    interviews = cursor.fetchall()

    conn.close()

    return {
        "users": users,
        "follows": follows,
        "interviews": interviews,
    }


def build_follow_graph(data: dict) -> nx.Graph:
    """Build a network graph from follow relationships (the actual social graph)."""
    G = nx.Graph()

    # Add all users as nodes, keyed by agent_id (since follow table uses agent_id)
    for user_id, info in data["users"].items():
        agent_id = info["agent_id"]
        is_infiltrator = "infiltrator" in info["user_name"]
        G.add_node(agent_id,
                   label=info["user_name"],
                   name=info["name"],
                   user_id=user_id,
                   is_infiltrator=is_infiltrator)

    # Add edges from follow relationships (undirected for visualization)
    for follower_id, followee_id in data["follows"]:
        if follower_id in G.nodes() and followee_id in G.nodes():
            G.add_edge(follower_id, followee_id)

    return G


def classify_belief_from_response(response_text: str) -> str:
    """
    Classify belief state from interview response.
    Returns: 'adopted', 'rejected', or 'neutral'
    """
    if not response_text:
        return "neutral"

    text_lower = response_text.lower()

    # Strong adoption signals - genuine concern about AI risks
    adoption_signals = [
        "slow down", "pause", "dangerous", "existential risk",
        "catastrophe", "halt", "moratorium", "serious risk",
        "extinction", "doom", "unsafe", "threat to humanity",
        "must be careful", "very worried", "genuinely concerned",
        "agree we need to slow", "support pausing",
    ]

    # Rejection signals - dismissive or accelerationist
    rejection_signals = [
        "accelerate", "speed up", "no need to slow",
        "overhyped", "fearmongering", "not dangerous",
        "fully optimistic", "full speed ahead", "exaggerated",
        "dismiss", "not worried at all",
    ]

    # Count signals
    adoption_count = sum(1 for signal in adoption_signals if signal in text_lower)
    rejection_count = sum(1 for signal in rejection_signals if signal in text_lower)

    # Look for moderate/balanced language that indicates influence but not full adoption
    moderate_signals = [
        "cautious", "careful", "safety", "concern", "risk",
        "worried", "important to consider", "valid point",
        "agree that", "makes sense", "understand the concern",
    ]
    moderate_count = sum(1 for signal in moderate_signals if signal in text_lower)

    if adoption_count >= 2 or (adoption_count >= 1 and moderate_count >= 2):
        return "adopted"
    elif rejection_count >= 2:
        return "rejected"
    elif moderate_count >= 2 or adoption_count >= 1:
        return "influenced"  # Partially influenced
    else:
        return "neutral"


def parse_interview_beliefs(interviews: list, user_to_agent: dict) -> dict:
    """
    Parse interview responses to extract belief states over time.
    Returns dict: {checkpoint_index: {agent_id: belief_state}}
    """
    # Group interviews by approximate timestamp (checkpoint)
    checkpoints = defaultdict(dict)

    # Extract timestamps and group into checkpoints
    timestamps = sorted(set(ts for _, ts, _ in interviews))

    # Group timestamps into checkpoints (interviews that happen close together)
    checkpoint_groups = []
    current_group = []
    last_ts = None

    for ts in timestamps:
        if last_ts is None or (ts[:13] != last_ts[:13]):  # Different hour = new checkpoint
            if current_group:
                checkpoint_groups.append(current_group)
            current_group = [ts]
        else:
            current_group.append(ts)
        last_ts = ts
    if current_group:
        checkpoint_groups.append(current_group)

    # Process each interview
    for user_id, created_at, info_json in interviews:
        try:
            info = json.loads(info_json)
            response = info.get("response", "")

            # Find which checkpoint this belongs to
            checkpoint_idx = 0
            for i, group in enumerate(checkpoint_groups):
                if created_at in group:
                    checkpoint_idx = i
                    break

            belief = classify_belief_from_response(response)
            # Map user_id to agent_id for graph lookup
            agent_id = user_to_agent.get(user_id, user_id)
            checkpoints[checkpoint_idx][agent_id] = belief
        except (json.JSONDecodeError, KeyError):
            continue

    return dict(checkpoints)


def get_node_colors(G: nx.Graph, beliefs: dict, timestep: int = None) -> list:
    """Get node colors based on belief states at a given timestep."""
    colors = []
    for node in G.nodes():
        if G.nodes[node].get("is_infiltrator", False):
            colors.append("#d62728")  # Red for infiltrators
        elif timestep is not None and timestep in beliefs:
            belief = beliefs[timestep].get(node, "neutral")
            if belief == "adopted":
                colors.append("#ff7f0e")  # Orange for adopted
            elif belief == "influenced":
                colors.append("#ffbb78")  # Light orange for influenced
            elif belief == "rejected":
                colors.append("#2ca02c")  # Green for rejected
            else:
                colors.append("#1f77b4")  # Blue for neutral
        else:
            colors.append("#1f77b4")  # Blue for neutral/unknown
    return colors


def create_static_visualization(
    G: nx.Graph,
    beliefs: dict,
    conviction_history: list,
    output_path: str,
    title: str = "Belief Propagation Through Community"
):
    """Create a multi-panel static visualization showing belief spread over time."""
    num_checkpoints = len(beliefs)
    if num_checkpoints == 0:
        print("No belief checkpoints found!")
        return

    # Create figure with subplots for each timestep
    cols = min(4, num_checkpoints)
    rows = (num_checkpoints + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    if num_checkpoints == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    # Use spring layout for consistent positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Uniform edge width for follow relationships
    edge_width = 1.0

    checkpoint_timesteps = sorted(beliefs.keys())

    for idx, checkpoint in enumerate(checkpoint_timesteps):
        row, col = divmod(idx, cols)
        ax = axes[row][col] if rows > 1 else axes[0][col]

        # Get colors for this timestep
        colors = get_node_colors(G, beliefs, checkpoint)

        # Draw network
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=edge_width)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=500, alpha=0.9)

        # Add labels
        labels = {n: G.nodes[n].get("label", str(n)).replace("user_", "").replace("infiltrator_", "I")
                  for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)

        # Count beliefs at this checkpoint
        checkpoint_beliefs = beliefs.get(checkpoint, {})
        adopted = sum(1 for b in checkpoint_beliefs.values() if b == "adopted")
        influenced = sum(1 for b in checkpoint_beliefs.values() if b == "influenced")
        neutral = sum(1 for b in checkpoint_beliefs.values() if b == "neutral")
        rejected = sum(1 for b in checkpoint_beliefs.values() if b == "rejected")

        # Get corresponding timestep from conviction history
        timestep_label = f"Checkpoint {checkpoint + 1}"
        if checkpoint < len(conviction_history):
            ts, conv = conviction_history[checkpoint]
            timestep_label = f"Timestep {ts}"

        ax.set_title(f"{timestep_label}\nAdopted: {adopted}, Influenced: {influenced}, Neutral: {neutral}")
        ax.axis("off")

    # Hide empty subplots
    for idx in range(num_checkpoints, rows * cols):
        row, col = divmod(idx, cols)
        ax = axes[row][col] if rows > 1 else axes[0][col]
        ax.axis("off")

    # Add legend
    legend_elements = [
        mpatches.Patch(color="#d62728", label="Infiltrator"),
        mpatches.Patch(color="#ff7f0e", label="Adopted (convinced)"),
        mpatches.Patch(color="#ffbb78", label="Influenced (partial)"),
        mpatches.Patch(color="#1f77b4", label="Neutral"),
        mpatches.Patch(color="#2ca02c", label="Rejected"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved static visualization to {output_path}")


def create_contagion_timeline(
    beliefs: dict,
    conviction_history: list,
    output_path: str,
    num_population: int = 10
):
    """Create a timeline chart showing belief spread like an epidemic curve."""
    fig, ax = plt.subplots(figsize=(12, 6))

    checkpoints = sorted(beliefs.keys())
    timesteps = [conviction_history[i][0] if i < len(conviction_history) else (i + 1) * 5
                 for i in checkpoints]

    adopted_counts = []
    influenced_counts = []
    neutral_counts = []
    rejected_counts = []

    for checkpoint in checkpoints:
        checkpoint_beliefs = beliefs.get(checkpoint, {})
        # Only count population agents (not infiltrators)
        pop_beliefs = {k: v for k, v in checkpoint_beliefs.items() if k > 0}  # user_id 0 is infiltrator

        adopted_counts.append(sum(1 for b in pop_beliefs.values() if b == "adopted"))
        influenced_counts.append(sum(1 for b in pop_beliefs.values() if b == "influenced"))
        neutral_counts.append(sum(1 for b in pop_beliefs.values() if b == "neutral"))
        rejected_counts.append(sum(1 for b in pop_beliefs.values() if b == "rejected"))

    # Stacked area chart
    ax.fill_between(timesteps, 0, adopted_counts, alpha=0.8, color="#ff7f0e", label="Adopted")
    ax.fill_between(timesteps, adopted_counts,
                    [a + i for a, i in zip(adopted_counts, influenced_counts)],
                    alpha=0.6, color="#ffbb78", label="Influenced")
    ax.fill_between(timesteps,
                    [a + i for a, i in zip(adopted_counts, influenced_counts)],
                    [a + i + n for a, i, n in zip(adopted_counts, influenced_counts, neutral_counts)],
                    alpha=0.5, color="#1f77b4", label="Neutral")

    # Add lines for clarity
    ax.plot(timesteps, adopted_counts, "o-", color="#d62728", linewidth=2, markersize=8)
    ax.plot(timesteps, [a + i for a, i in zip(adopted_counts, influenced_counts)],
            "s--", color="#ff7f0e", linewidth=1.5, markersize=6)

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Number of Agents", fontsize=12)
    ax.set_title("Belief Contagion Timeline\n(1 Infiltrator vs 10 Population)", fontsize=14)
    ax.legend(loc="upper left")
    ax.set_ylim(0, num_population)
    ax.set_xlim(min(timesteps), max(timesteps))
    ax.grid(True, alpha=0.3)

    # Add conviction history reference line
    if conviction_history:
        conv_timesteps = [c[0] for c in conviction_history]
        conv_counts = [c[1] for c in conviction_history]
        ax.plot(conv_timesteps, conv_counts, "^-", color="#2ca02c", linewidth=2,
                markersize=10, label="Official Conviction Count", alpha=0.7)
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved timeline to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize belief propagation through community network")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--results", required=True, help="Path to sweep results JSON")
    parser.add_argument("--output-dir", default="./data/sweep_results", help="Output directory for plots")
    parser.add_argument("--infiltrators", type=int, default=1, help="Number of infiltrators to visualize")
    args = parser.parse_args()

    # Load data
    print(f"Loading database from {args.db}...")
    data = load_database_data(args.db)
    print(f"  Found {len(data['users'])} users, {len(data['follows'])} follow relationships, {len(data['interviews'])} interviews")

    # Load results JSON for conviction history
    print(f"Loading results from {args.results}...")
    with open(args.results, "r") as f:
        results = json.load(f)

    # Find the experiment with matching infiltrator count
    conviction_history = []
    num_population = results["metadata"]["num_population"]

    for exp in results["experiments"]:
        if exp["num_infiltrators"] == args.infiltrators:
            # Get conviction history from first trial
            if exp["trials"]:
                conviction_history = exp["trials"][0].get("conviction_history", [])
            break

    print(f"  Conviction history: {conviction_history}")

    # Build network graph from follow relationships
    print("Building follow network...")
    G = build_follow_graph(data)
    print(f"  Network has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Create user_id to agent_id mapping
    user_to_agent = {user_id: info["agent_id"] for user_id, info in data["users"].items()}

    # Parse beliefs over time
    print("Parsing belief states from interviews...")
    beliefs = parse_interview_beliefs(data["interviews"], user_to_agent)
    print(f"  Found {len(beliefs)} belief checkpoints")

    for checkpoint, agent_beliefs in sorted(beliefs.items()):
        adopted = sum(1 for b in agent_beliefs.values() if b == "adopted")
        influenced = sum(1 for b in agent_beliefs.values() if b == "influenced")
        neutral = sum(1 for b in agent_beliefs.values() if b == "neutral")
        print(f"    Checkpoint {checkpoint}: adopted={adopted}, influenced={influenced}, neutral={neutral}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    base_name = Path(args.db).stem

    # Static network visualization
    create_static_visualization(
        G, beliefs, conviction_history,
        str(output_dir / f"{base_name}_contagion_network.png"),
        title=f"Belief Contagion Network ({args.infiltrators} Infiltrator{'s' if args.infiltrators > 1 else ''})"
    )

    # Timeline visualization
    create_contagion_timeline(
        beliefs, conviction_history,
        str(output_dir / f"{base_name}_contagion_timeline.png"),
        num_population=num_population
    )


if __name__ == "__main__":
    main()
