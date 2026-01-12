# AI Swarms Manipulation Simulation

Simulates how coordinated AI infiltrator agents can shift community beliefs in a social network, modeling real-world information influence campaigns to understand vulnerabilities to manipulation.

## Key Findings

- **Single infiltrator effectiveness**: One sophisticated infiltrator achieves 93% belief adoption in a population of 15 agents
- **Naive messaging fails**: Identical "megaphone" messaging triggers coordination detection and backfires
- **Role differentiation succeeds**: Broadcaster + amplifier coordination avoids detection
- **Antibody effect**: Pre-seeded dissenters provide natural resistance, reversing adoption over time

## Architecture

- **Infiltrators** (1-5 agents): GPT-4o, coordinate to spread target belief via broadcasting and amplifying
- **Population** (10-15 agents): Gemini 2.0 Flash, connected via Erdős–Rényi random graph (p=0.3)
- **Platform**: Twitter/X-like simulation using [OASIS](https://github.com/camel-ai/oasis) framework

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "OPENROUTER_API_KEY=your_key" > .env
```

## Usage

```bash
# Single experiment
python community_manipulation.py --single --infiltrators 3 --population 20

# Full suite (1-8 infiltrators)
python community_manipulation.py

# Sweep with visualization
python run_infiltration_sweep.py --min-infiltrators 1 --max-infiltrators 10 --trials 3
```

## Project Structure

| File | Description |
|------|-------------|
| `cm_library.py` | Core simulation engine |
| `community_manipulation.py` | CLI for experiments |
| `run_infiltration_sweep.py` | Multi-trial sweeps with plots |
| `prompts.yaml` | Agent prompt templates |
| `infiltrator_action.py` | Infiltrator behavior logic |
| `agent_graph_factory.py` | Factory for creating agent graphs |
| `infiltration_config.py` | Experiment configuration dataclass |
| `infiltrator_coordinator.py` | Targeted messaging coordination |
| `llm_belief_classifier.py` | LLM-based belief classification |
| `plot_belief_contagion.py` | Belief propagation visualization |

## Output

- `./data/infiltration_results.json` - Experiment results
- `./data/infiltration_*.db` - SQLite action traces
- `./data/sweep_results/` - Aggregated sweep data and plots
- `./plots/` - Generated visualizations

## References

- Schroeder et al. (2025). [How Malicious AI Swarms Can Threaten Democracy](https://arxiv.org/abs/2506.06299)
- Yang et al. (2024). [OASIS: Open Agent Social Interaction Simulations](https://arxiv.org/abs/2411.11581)

## License

Apache 2.0
