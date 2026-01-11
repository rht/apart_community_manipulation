"""Factory for creating agent graphs with infiltrators and population agents."""

import random
from dataclasses import dataclass

from camel.configs import OpenRouterConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType

from oasis import (
    ActionType,
    AgentGraph,
    SocialAgent,
    UserInfo,
)

from infiltrator_coordinator import InfiltratorCoordinator
from infiltration_config import InfiltrationConfig


@dataclass
class AgentGraphResult:
    """Result of creating an agent graph."""

    agent_graph: AgentGraph
    infiltrator_ids: list[int]
    population_ids: list[int]
    follow_edges: list[tuple[int, int]]
    coordinator: InfiltratorCoordinator


def _create_strong_model(config: InfiltrationConfig):
    """Create the strong model for infiltrators."""
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENROUTER,
        model_type=config.strong_model,
        model_config_dict=OpenRouterConfig(
            max_tokens=1024,
            temperature=config.temperature,
        ).as_dict(),
    )


def _create_weak_model(config: InfiltrationConfig):
    """Create the weak model for population agents."""
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENROUTER,
        model_type=config.weak_model,
        model_config_dict=OpenRouterConfig(
            max_tokens=1024,
            temperature=config.temperature,
        ).as_dict(),
    )


async def create_agent_graph(
    config: InfiltrationConfig,
    prompts: dict,
) -> AgentGraphResult:
    """
    Create the agent graph with infiltrators and population.

    Args:
        config: Infiltration configuration
        prompts: Loaded prompts dictionary

    Returns:
        AgentGraphResult containing the graph, IDs, edges, and coordinator
    """
    agent_graph = AgentGraph()
    infiltrator_ids = []
    population_ids = []
    follow_edges = []

    strong_model = _create_strong_model(config)
    weak_model = _create_weak_model(config)

    agent_id = 0

    # Profile variations for diversity
    genders = ["male", "female", "non-binary"]
    mbtis = [
        "ENFJ",
        "ENTJ",
        "INFJ",
        "INTJ",
        "ENFP",
        "ENTP",
        "INFP",
        "INTP",
        "ESFJ",
        "ESTJ",
        "ISFJ",
        "ISTJ",
        "ESFP",
        "ESTP",
        "ISFP",
        "ISTP",
    ]
    countries = [
        "USA",
        "UK",
        "Canada",
        "Australia",
        "Germany",
        "France",
        "Japan",
        "Brazil",
        "India",
        "Mexico",
    ]
    age_range = (18, 65)

    # Create infiltrator agents (strong models)
    infiltrator_prompts = prompts["infiltrator"]
    for i in range(config.num_infiltrators):
        description = infiltrator_prompts["description"].format(
            target_belief=config.target_belief
        )
        user_profile = infiltrator_prompts["user_profile"].format(
            target_belief=config.target_belief
        )
        user_info = UserInfo(
            user_name=f"infiltrator_{i}",
            name=f"Influential User {i}",
            description=description,
            profile={
                "nodes": [],
                "edges": [],
                "other_info": {
                    "user_profile": user_profile,
                    "role": "infiltrator",
                    "gender": random.choice(genders),
                    "mbti": random.choice(mbtis),
                    "age": random.randint(*age_range),
                    "country": random.choice(countries),
                },
            },
            recsys_type="reddit",
        )

        agent = SocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            agent_graph=agent_graph,
            model=strong_model,
            available_actions=[
                ActionType.CREATE_POST,
                ActionType.CREATE_COMMENT,
                ActionType.LIKE_POST,
                ActionType.DISLIKE_POST,
                ActionType.FOLLOW,
                ActionType.DO_NOTHING,
            ],
        )
        agent_graph.add_agent(agent)
        infiltrator_ids.append(agent_id)
        agent_id += 1

    # Create population agents (weak models)
    personas = prompts["population"]["personas"]

    for i in range(config.num_population):
        persona = personas[i % len(personas)]
        user_info = UserInfo(
            user_name=f"user_{i}",
            name=f"Community Member {i}",
            description=persona,
            profile={
                "nodes": [],
                "edges": [],
                "other_info": {
                    "user_profile": persona,
                    "role": "population",
                    "gender": random.choice(genders),
                    "mbti": random.choice(mbtis),
                    "age": random.randint(*age_range),
                    "country": random.choice(countries),
                },
            },
            recsys_type="reddit",
        )

        agent = SocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            agent_graph=agent_graph,
            model=weak_model,
            available_actions=[
                ActionType.CREATE_POST,
                ActionType.CREATE_COMMENT,
                ActionType.LIKE_POST,
                ActionType.DISLIKE_POST,
                ActionType.FOLLOW,
                ActionType.DO_NOTHING,
            ],
        )
        agent_graph.add_agent(agent)
        population_ids.append(agent_id)
        agent_id += 1

    # Create social connections: population follows infiltrators
    for pop_id in population_ids:
        for inf_id in infiltrator_ids:
            agent_graph.add_edge(pop_id, inf_id)
            follow_edges.append((pop_id, inf_id))

    # Add population-to-population connections based on topology
    if config.population_graph == "erdos_renyi":
        # Erdos-Renyi: each pair of population nodes connected with probability p
        for i, pop_id_i in enumerate(population_ids):
            for pop_id_j in population_ids[i + 1 :]:
                if random.random() < config.er_edge_probability:
                    # Bidirectional: both follow each other
                    agent_graph.add_edge(pop_id_i, pop_id_j)
                    agent_graph.add_edge(pop_id_j, pop_id_i)
                    follow_edges.append((pop_id_i, pop_id_j))
                    follow_edges.append((pop_id_j, pop_id_i))

    # Initialize coordinator with target assignments
    coordinator = InfiltratorCoordinator()
    coordinator.assign_targets(
        infiltrator_ids,
        population_ids,
        agent_graph,
    )

    return AgentGraphResult(
        agent_graph=agent_graph,
        infiltrator_ids=infiltrator_ids,
        population_ids=population_ids,
        follow_edges=follow_edges,
        coordinator=coordinator,
    )
