from app.config import settings

def get_reward_for_event(event_type: str) -> float:
    """Return numeric reward for the given interaction event type."""
    rewards = settings.rewards
    mapping = {
        "purchase": rewards.purchase,
        "cart_add": rewards.cart_add,
        "wishlist": rewards.wishlist,
        "click": rewards.click,
        "skip": rewards.skip,
        "cart_remove": rewards.cart_remove,
        "dislike": rewards.dislike,
        "return": rewards.return_item,
        "impression": 0.0
    }
    return mapping.get(event_type, 0.0)

