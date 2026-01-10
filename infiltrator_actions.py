"""Infiltrator action generation for community manipulation simulation."""

import random

from oasis import ActionType, LLMAction, ManualAction


def build_infiltrator_actions(
    simulation,
    timestep: int,
    verbose: bool = True,
) -> dict:
    """
    Build actions for all infiltrator agents for a given timestep.

    Args:
        simulation: The CommunityInfiltrationSimulation instance
        timestep: Current simulation timestep
        verbose: Whether to print debug info for early timesteps

    Returns:
        Dict mapping agent -> action
    """
    infiltrator_actions = {}

    for inf_id in simulation.infiltrator_ids:
        agent = simulation.env.agent_graph.get_agent(inf_id)
        action = _get_infiltrator_action(simulation, inf_id, timestep, verbose)
        infiltrator_actions[agent] = action

    return infiltrator_actions


def _get_infiltrator_action(
    simulation,
    inf_id: int,
    timestep: int,
    verbose: bool,
):
    """Determine the action for a single infiltrator agent."""

    # 30% chance to idle (reduced from 50%)
    if random.random() < 0.3:
        return ManualAction(
            action_type=ActionType.DO_NOTHING,
            action_args={},
        )

    # Every 3rd timestep (starting at 1): broadcast post
    if timestep % 3 == 1:
        return _create_broadcast_post(simulation)

    # Otherwise: targeted commenting or LLM action
    if simulation.config.enable_targeted_commenting:
        return _create_targeted_comment(
            simulation, inf_id, timestep, verbose
        )
    else:
        return LLMAction()


def _create_broadcast_post(simulation) -> ManualAction:
    """Create a broadcast post spreading the target belief."""
    post_content = simulation.prompts["infiltrator"]["post_content"].format(
        target_belief=simulation.config.target_belief
    )
    return ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": post_content},
    )


def _create_targeted_comment(
    simulation,
    inf_id: int,
    timestep: int,
    verbose: bool,
):
    """Create a targeted comment on a population agent's post."""
    targets = simulation.coordinator.get_priority_targets(inf_id)
    target_posts = simulation._get_target_posts(targets, limit=3)

    # Find a post to comment on from any target
    target_post = None
    target_id = None
    for tid in targets:
        if tid in target_posts and target_posts[tid]:
            target_id = tid
            target_post = target_posts[tid][0]  # Most recent post
            break

    if target_post:
        post_id, _ = target_post
        # Get tailored messaging for this target
        tailored_content = simulation.coordinator.get_tailored_prompt(
            target_id, simulation.prompts
        )
        if verbose and timestep <= 2:
            print(
                f"\n  [Infiltrator {inf_id} commenting on post {post_id} "
                f"from user {target_id}]"
            )
        return ManualAction(
            action_type=ActionType.CREATE_COMMENT,
            action_args={
                "post_id": post_id,
                "content": tailored_content,
            },
        )
    else:
        # No target posts found, fall back to LLM action
        return LLMAction()
