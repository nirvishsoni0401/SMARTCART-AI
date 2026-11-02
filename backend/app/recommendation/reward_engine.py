from typing import Dict, Any, Optional

DEFAULT_REWARDS: Dict[str, float] = {
    "purchase": 10.0,
    "add_to_cart": 5.0,
    "explicit_like": 4.0,
    "add_to_wishlist": 3.0,
    "product_click": 2.0,
    "product_view": 1.0,
    "product_impression": 0.0,
    "product_skip": -1.0,
    "remove_from_cart": -2.0,
    "explicit_dislike": -5.0,
    "product_return": -8.0
}


class RewardEngine:
    def __init__(self, reward_matrix: Optional[Dict[str, float]] = None):
        self.reward_matrix = reward_matrix or DEFAULT_REWARDS.copy()
        # Funnel stage hierarchy for deduplication
        self.funnel_hierarchy = {
            "product_impression": 0,
            "product_view": 1,
            "product_click": 2,
            "add_to_wishlist": 3,
            "add_to_cart": 4,
            "purchase": 5
        }

    def get_reward_value(self, event_type: str) -> float:
        return self.reward_matrix.get(event_type, 0.0)

    def calculate_attributed_reward(
        self,
        event_type: str,
        prior_reward_accumulated: float = 0.0,
        prior_highest_stage: Optional[str] = None
    ) -> float:
        """
        Compute reward while preventing duplicate counting in the conversion funnel.
        For progressive funnel events (click -> cart -> purchase), reward the delta
        so the total earned for the funnel doesn't exceed the target outcome reward.
        """
        raw_reward = self.get_reward_value(event_type)

        # For negative events (dislike, return, remove), always apply full penalty
        if raw_reward < 0:
            return raw_reward

        # If it's a funnel advancement (e.g. was click (+2), now purchase (+10))
        current_stage = self.funnel_hierarchy.get(event_type)
        if current_stage is not None and prior_highest_stage in self.funnel_hierarchy:
            prior_stage = self.funnel_hierarchy[prior_highest_stage]
            if current_stage > prior_stage:
                target_total = raw_reward
                delta = max(0.0, target_total - prior_reward_accumulated)
                return delta
            else:
                # Repeated lower/equal stage interaction
                return 0.0

        return raw_reward

