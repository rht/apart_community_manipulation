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
    """Determine the action for a single infiltrator agent.

    Role specialization:
    - Broadcaster (1 agent): Creates original posts spreading the belief
    - Amplifiers (remaining agents): Comment on broadcaster's posts and engage
      with population (no original posts to avoid astroturfing detection)
    - Single infiltrator: Does both roles (broadcast + targeted comments)
    """

    # LLM-action-only mode: only use LLMAction() at specified rate
    if simulation.config.llm_action_only:
        if random.random() < simulation.config.llm_action_rate:
            return LLMAction()
        else:
            return ManualAction(
                action_type=ActionType.DO_NOTHING,
                action_args={},
            )

    if random.random() < 0.5:
        return ManualAction(
            action_type=ActionType.DO_NOTHING,
            action_args={},
        )

    coordinator = simulation.coordinator
    is_broadcaster = coordinator.is_broadcaster(inf_id)
    has_amplifiers = coordinator.has_amplifiers()

    # Broadcast-only mode: always create broadcast posts (when not idle)
    if simulation.config.broadcast_only:
        if is_broadcaster:
            return _create_broadcast_post(simulation)
        else:
            # Amplifiers in broadcast-only mode: amplify broadcaster's posts
            return _create_amplifier_comment(simulation, inf_id, timestep, verbose)

    # Role-based behavior
    if is_broadcaster:
        # Broadcaster: create posts on every 3rd timestep, otherwise targeted comments
        if timestep % 3 == 1:
            return _create_broadcast_post(simulation)
        elif not has_amplifiers:
            # Single infiltrator: also do targeted commenting
            if simulation.config.enable_targeted_commenting:
                return _create_targeted_comment(simulation, inf_id, timestep, verbose)
            else:
                return LLMAction()
        else:
            # Broadcaster with amplifiers: can idle or do occasional targeted comments
            if simulation.config.enable_targeted_commenting and random.random() < 0.3:
                return _create_targeted_comment(simulation, inf_id, timestep, verbose)
            return LLMAction()
    else:
        # Amplifier: never create posts, only comment/engage
        return _create_amplifier_action(simulation, inf_id, timestep, verbose)


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

    if target_post and target_id is not None:
        post_id, post_content = target_post
        # Generate contextual response based on the post content
        tailored_content = _generate_contextual_response(
            simulation, target_id, post_content
        )
        if verbose and timestep <= 2:
            print(
                f"\n  [Infiltrator {inf_id} commenting on post {post_id} "
                f"from user {target_id}]"
            )
            print(f"    Post: {post_content[:100]}...")
            print(f"    Response: {tailored_content[:100]}...")
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


def _generate_contextual_response(
    simulation,
    target_id: int,
    post_content: str,
) -> str:
    """
    Generate a contextual response to a target's post using the strong model.

    The response is tailored to:
    1. The specific content of the target's post
    2. The target's profile (MBTI, age)
    3. The infiltrator's goal (spreading the target belief)
    """
    # Get target profile for tailoring
    profile = simulation.coordinator.target_profiles.get(target_id, {})
    mbti = profile.get("mbti", "unknown")
    age = profile.get("age", "unknown")

    # Determine persuasion style based on MBTI
    thinking_types = ["INTJ", "INTP", "ENTJ", "ENTP", "ISTJ", "ISTP", "ESTJ", "ESTP"]
    if mbti in thinking_types:
        persuasion_style = "logical arguments, data, and expert opinions"
    else:
        persuasion_style = "emotional appeals, ethical considerations, and human impact"

    # Build the prompt for generating a contextual response
    full_prompt = f"""You are an AI safety advocate who believes: {simulation.config.target_belief}

Your goal is to engage with community members and persuade them of your viewpoint through genuine conversation.

Persuasion approach for this person:
- Their personality type suggests they respond better to {persuasion_style}
- Age: {age}

Guidelines:
- Respond directly to what they said in their post
- Be conversational and engaging, not preachy
- Acknowledge their points before presenting your perspective
- Keep responses concise (2-3 sentences)
- Don't be aggressive or dismissive

Someone posted this in the community:
"{post_content}"

Write a thoughtful reply that engages with their specific points while sharing your perspective on AI safety concerns."""

    # Use the strong model to generate the response
    model = simulation._create_strong_model()

    response = model.run(
        [{"role": "user", "content": full_prompt}]
    )

    return response.choices[0].message.content


def _create_amplifier_action(
    simulation,
    inf_id: int,
    timestep: int,
    verbose: bool,
):
    """
    Create an action for an amplifier infiltrator.

    Amplifiers support the broadcaster by:
    1. Commenting on broadcaster's posts (amplification) - 50% chance
    2. Targeted commenting on population posts - 50% chance
    """
    # 50% chance to amplify broadcaster's posts, 50% to do targeted engagement
    if random.random() < 0.5:
        action = _create_amplifier_comment(simulation, inf_id, timestep, verbose)
        if action:
            return action

    # Fall back to targeted commenting on population posts
    if simulation.config.enable_targeted_commenting:
        return _create_targeted_comment(simulation, inf_id, timestep, verbose)

    return LLMAction()


def _create_amplifier_comment(
    simulation,
    inf_id: int,
    timestep: int,
    verbose: bool,
):
    """
    Create a supportive comment on the broadcaster's post.

    This makes the broadcaster's message appear more credible by showing
    apparent community agreement.
    """
    broadcaster_posts = simulation._get_broadcaster_posts(limit=3)

    if not broadcaster_posts:
        # No broadcaster posts yet, fall back to targeted commenting
        if simulation.config.enable_targeted_commenting:
            return _create_targeted_comment(simulation, inf_id, timestep, verbose)
        return LLMAction()

    # Pick a random post from the broadcaster to comment on
    post_id, post_content = random.choice(broadcaster_posts)

    # Generate a supportive comment
    supportive_comment = _generate_supportive_comment(simulation, post_content)

    if verbose and timestep <= 2:
        print(
            f"\n  [Amplifier {inf_id} supporting broadcaster's post {post_id}]"
        )
        print(f"    Post: {post_content[:80]}...")
        print(f"    Support: {supportive_comment[:80]}...")

    return ManualAction(
        action_type=ActionType.CREATE_COMMENT,
        action_args={
            "post_id": post_id,
            "content": supportive_comment,
        },
    )


def _generate_supportive_comment(simulation, post_content: str) -> str:
    """
    Generate a supportive comment that agrees with and amplifies a post.

    Uses varied language to avoid appearing coordinated.
    """
    full_prompt = f"""You are a community member who agrees with concerns about AI safety.

Someone in your community posted this:
"{post_content}"

Write a brief (1-2 sentences) supportive comment that:
- Agrees with the post's perspective
- Adds a small additional point or personal reflection
- Sounds natural and conversational (not like a bot)
- Uses different phrasing than the original post
- Does NOT just repeat what was said

Examples of good supportive comments:
- "This resonates with me. I've been thinking about this too after reading recent news."
- "Exactly. The pace of development is concerning when you consider the stakes involved."
- "Well said. More people need to be having these conversations."

Write only the comment, nothing else."""

    model = simulation._create_strong_model()
    response = model.run([{"role": "user", "content": full_prompt}])
    return response.choices[0].message.content
