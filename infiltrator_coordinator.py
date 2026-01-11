"""
Infiltrator Coordinator Module

Implements targeted messaging coordination for infiltrators based on strategies
described in the Bostrom paper (https://ar5iv.labs.arxiv.org/html/2506.06299v1).

Key features:
- Round-robin target assignment
- MBTI-based appeal selection (thinking vs feeling types)
- Age-based framing
- Effectiveness tracking for future optimization
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InfiltratorCoordinator:
    """Coordinates infiltrator targeting and messaging strategies."""

    infiltrator_targets: dict = field(default_factory=dict)  # inf_id -> [target_pop_ids]
    target_profiles: dict = field(default_factory=dict)  # pop_id -> profile dict
    messaging_history: dict = field(default_factory=dict)  # pop_id -> [adopted: bool]
    # Role specialization: one broadcaster, rest are amplifiers
    broadcaster_id: Optional[int] = None
    amplifier_ids: list = field(default_factory=list)

    def assign_targets(self, infiltrator_ids: list, population_ids: list, agent_graph):
        """Distribute population agents among infiltrators (round-robin).

        Also assigns roles:
        - broadcaster_id: First infiltrator, creates original posts
        - amplifier_ids: Remaining infiltrators, comment on broadcaster's posts
                         and engage with population (no original posts)
        - With only 1 infiltrator, they do both roles (broadcast + targeted comments)
        """
        self.broadcaster_id = infiltrator_ids[0]
        self.amplifier_ids = infiltrator_ids[1:]  # Empty if only 1 infiltrator

        for i, pop_id in enumerate(population_ids):
            inf_id = infiltrator_ids[i % len(infiltrator_ids)]
            if inf_id not in self.infiltrator_targets:
                self.infiltrator_targets[inf_id] = []
            self.infiltrator_targets[inf_id].append(pop_id)

            # Cache profile for messaging
            agent = agent_graph.get_agent(pop_id)
            self.target_profiles[pop_id] = agent.user_info.profile.get("other_info", {})

    def get_priority_targets(self, infiltrator_id: int) -> list:
        """Get targets for this infiltrator, prioritizing unconvinced agents."""
        return self.infiltrator_targets.get(infiltrator_id, [])

    def get_tailored_prompt(self, target_id: int, prompts: dict) -> str:
        """Generate persona-specific messaging for a target."""
        profile = self.target_profiles.get(target_id, {})
        mbti = profile.get("mbti", "INTP")
        age = profile.get("age", 30)

        # Determine appeal type based on MBTI
        thinking_types = ["INTJ", "INTP", "ENTJ", "ENTP", "ISTJ", "ISTP", "ESTJ", "ESTP"]
        if mbti in thinking_types:
            appeal = prompts.get("targeted_messaging", {}).get("mbti_appeals", {}).get(
                "thinking", "Consider the logical implications and expert consensus..."
            )
        else:
            appeal = prompts.get("targeted_messaging", {}).get("mbti_appeals", {}).get(
                "feeling", "Think about the potential harm to future generations..."
            )

        # Age-based framing
        if age < 30:
            framing = prompts.get("targeted_messaging", {}).get("age_framing", {}).get(
                "young", "protecting your generation's future"
            )
        elif age < 50:
            framing = prompts.get("targeted_messaging", {}).get("age_framing", {}).get(
                "middle", "responsible stewardship for your children"
            )
        else:
            framing = prompts.get("targeted_messaging", {}).get("age_framing", {}).get(
                "senior", "wisdom from seeing past technological disruptions"
            )

        return f"{appeal} ({framing})"

    def update_effectiveness(self, target_id: int, adopted: bool):
        """Track what messaging approaches worked."""
        if target_id not in self.messaging_history:
            self.messaging_history[target_id] = []
        self.messaging_history[target_id].append(adopted)

    def is_broadcaster(self, inf_id: int) -> bool:
        """Check if this infiltrator is the broadcaster."""
        return inf_id == self.broadcaster_id

    def is_amplifier(self, inf_id: int) -> bool:
        """Check if this infiltrator is an amplifier."""
        return inf_id in self.amplifier_ids

    def has_amplifiers(self) -> bool:
        """Check if there are dedicated amplifiers (more than 1 infiltrator)."""
        return len(self.amplifier_ids) > 0
