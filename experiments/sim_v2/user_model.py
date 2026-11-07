"""
SmartCart AI — Calibrated Simulator v2: User Model & Personas
Defines 10 realistic customer personas across the 6 active catalog departments:
  Beauty, Books, Electronics, Fashion, Home & Kitchen, Sports.

Includes:
  - Category affinities, brand affinities, budget/price sensitivity, discount sensitivity, novelty preference.
  - User instance sampling with Gaussian perturbation around persona archetypes.
  - Arrival process: new users (cold start) vs returning users (with session/interaction history).
  - Repeat intent / re-engagement effect (recency bonus for previously engaged products).
  - Drift models: Abrupt drift, Gradual drift, and Stationary (no drift).
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import math
import numpy as np

CATEGORIES = [
    "Electronics",
    "Fashion",
    "Home & Kitchen",
    "Beauty",
    "Sports",
    "Books"
]

# 10 Latent Persona Archetypes
PERSONA_ARCHETYPES = [
    {
        "id": 0,
        "name": "Tech Gadgeteer",
        "category_affinities": {"Electronics": 0.75, "Sports": 0.10, "Home & Kitchen": 0.05, "Books": 0.05, "Fashion": 0.03, "Beauty": 0.02},
        "preferred_brands": {"Apple", "Samsung", "Dell", "Sony"},
        "budget_mean": 38000.0,
        "price_sensitivity": 0.30,
        "discount_sensitivity": 0.35,
        "novelty_preference": 0.70,
    },
    {
        "id": 1,
        "name": "Fashion Trendsetter",
        "category_affinities": {"Fashion": 0.80, "Beauty": 0.12, "Electronics": 0.03, "Home & Kitchen": 0.02, "Sports": 0.02, "Books": 0.01},
        "preferred_brands": {"Rolex", "Fashion Shades", "Gucci", "Zara", "Nike"},
        "budget_mean": 45000.0,
        "price_sensitivity": 0.25,
        "discount_sensitivity": 0.60,
        "novelty_preference": 0.85,
    },
    {
        "id": 2,
        "name": "Avid Bookworm",
        "category_affinities": {"Books": 0.85, "Home & Kitchen": 0.08, "Beauty": 0.02, "Electronics": 0.02, "Fashion": 0.02, "Sports": 0.01},
        "preferred_brands": {"Penguin India", "HarperCollins", "Bloomsbury", "Pan Macmillan", "Rupa Publications"},
        "budget_mean": 850.0,
        "price_sensitivity": 0.40,
        "discount_sensitivity": 0.30,
        "novelty_preference": 0.65,
    },
    {
        "id": 3,
        "name": "Home & Kitchen Organizer",
        "category_affinities": {"Home & Kitchen": 0.75, "Electronics": 0.12, "Books": 0.05, "Beauty": 0.04, "Fashion": 0.02, "Sports": 0.02},
        "preferred_brands": {"Unbranded", "Philips", "IKEA"},
        "budget_mean": 14000.0,
        "price_sensitivity": 0.50,
        "discount_sensitivity": 0.65,
        "novelty_preference": 0.40,
    },
    {
        "id": 4,
        "name": "Active Athlete",
        "category_affinities": {"Sports": 0.75, "Fashion": 0.15, "Electronics": 0.05, "Beauty": 0.02, "Home & Kitchen": 0.02, "Books": 0.01},
        "preferred_brands": {"Nike", "Adidas", "Puma", "Decathlon"},
        "budget_mean": 3200.0,
        "price_sensitivity": 0.45,
        "discount_sensitivity": 0.50,
        "novelty_preference": 0.55,
    },
    {
        "id": 5,
        "name": "Clean Beauty Lover",
        "category_affinities": {"Beauty": 0.80, "Fashion": 0.10, "Home & Kitchen": 0.05, "Books": 0.02, "Sports": 0.02, "Electronics": 0.01},
        "preferred_brands": {"L'Oreal", "Nykaa", "Forest Essentials", "Unbranded"},
        "budget_mean": 3500.0,
        "price_sensitivity": 0.35,
        "discount_sensitivity": 0.55,
        "novelty_preference": 0.75,
    },
    {
        "id": 6,
        "name": "Budget Bargain Hunter",
        "category_affinities": {"Fashion": 0.25, "Electronics": 0.20, "Home & Kitchen": 0.20, "Books": 0.15, "Beauty": 0.10, "Sports": 0.10},
        "preferred_brands": set(),
        "budget_mean": 1200.0,
        "price_sensitivity": 0.90,
        "discount_sensitivity": 0.95,
        "novelty_preference": 0.30,
    },
    {
        "id": 7,
        "name": "Luxury Connoisseur",
        "category_affinities": {"Electronics": 0.50, "Fashion": 0.40, "Beauty": 0.05, "Home & Kitchen": 0.03, "Sports": 0.01, "Books": 0.01},
        "preferred_brands": {"Apple", "Rolex", "Samsung"},
        "budget_mean": 120000.0,
        "price_sensitivity": 0.10,
        "discount_sensitivity": 0.10,
        "novelty_preference": 0.60,
    },
    {
        "id": 8,
        "name": "Student / Tech Learner",
        "category_affinities": {"Books": 0.65, "Electronics": 0.25, "Sports": 0.04, "Fashion": 0.03, "Home & Kitchen": 0.02, "Beauty": 0.01},
        "preferred_brands": {"Packt", "O'Reilly", "S. Chand", "Jaico"},
        "budget_mean": 700.0,
        "price_sensitivity": 0.85,
        "discount_sensitivity": 0.80,
        "novelty_preference": 0.50,
    },
    {
        "id": 9,
        "name": "Casual Family Shopper",
        "category_affinities": {"Home & Kitchen": 0.40, "Books": 0.25, "Beauty": 0.15, "Fashion": 0.10, "Sports": 0.05, "Electronics": 0.05},
        "preferred_brands": set(),
        "budget_mean": 4500.0,
        "price_sensitivity": 0.60,
        "discount_sensitivity": 0.70,
        "novelty_preference": 0.45,
    }
]


class UserInstance:
    """Represents a specific simulated shopper with personal state."""

    def __init__(
        self,
        user_id: int,
        persona: Dict[str, Any],
        rng: np.random.Generator
    ):
        self.user_id = user_id
        self.persona_id = persona["id"]
        self.persona_name = persona["name"]

        # Sample personal affinities around persona mean
        raw_aff = {}
        for cat in CATEGORIES:
            base = persona["category_affinities"].get(cat, 0.05)
            # Log-normal perturbation
            noise = math.exp(rng.normal(0.0, 0.20))
            raw_aff[cat] = base * noise
        total_aff = sum(raw_aff.values())
        self.category_affinities = {cat: raw_aff[cat] / total_aff for cat in CATEGORIES}

        # Personal budget (log-normal around persona budget)
        budget_noise = math.exp(rng.normal(0.0, 0.25))
        self.budget = float(persona["budget_mean"] * budget_noise)

        # Personal sensitivities
        self.price_sensitivity = float(np.clip(persona["price_sensitivity"] + rng.normal(0.0, 0.05), 0.05, 1.0))
        self.discount_sensitivity = float(np.clip(persona["discount_sensitivity"] + rng.normal(0.0, 0.05), 0.05, 1.0))
        self.novelty_preference = float(np.clip(persona["novelty_preference"] + rng.normal(0.0, 0.05), 0.05, 1.0))
        self.preferred_brands = set(persona["preferred_brands"])

        # Latent style preference (3D unit vector for hidden style factor)
        style_raw = rng.normal(0.0, 1.0, size=3)
        self.latent_style = style_raw / max(1e-8, float(np.linalg.norm(style_raw)))

        # Dynamic state
        self.history: List[Tuple[int, int, str]] = []  # (product_id, round, event_type)
        self.cart: Set[int] = set()
        self.wishlist: Set[int] = set()
        self.purchased: Set[int] = set()
        self.total_sessions = 0
        self.last_round_seen = -1

    def get_viewed_product_ids(self) -> Set[int]:
        return {item[0] for item in self.history}

    def apply_drift(self, drift_ratio: float, drifted_archetype: Dict[str, Any]):
        """Interpolates preferences toward a drifted archetype (0 = original, 1 = fully drifted)."""
        if drift_ratio <= 0.0:
            return
        w_orig = 1.0 - drift_ratio
        w_new = drift_ratio

        # Interpolate category affinities
        for cat in CATEGORIES:
            target_aff = drifted_archetype["category_affinities"].get(cat, 0.05)
            self.category_affinities[cat] = w_orig * self.category_affinities[cat] + w_new * target_aff
        total = sum(self.category_affinities.values())
        for cat in CATEGORIES:
            self.category_affinities[cat] /= total

        # Interpolate budget
        self.budget = w_orig * self.budget + w_new * (drifted_archetype["budget_mean"])

        # Invert/shift style preferences under full drift
        if drift_ratio >= 1.0:
            self.latent_style = -self.latent_style


class UserManager:
    """Manages the pool of active and arriving users across the simulation horizon."""

    def __init__(
        self,
        n_initial_users: int = 150,
        p_new_arrival: float = 0.25,
        drift_mode: str = "none",  # "none", "abrupt", "gradual"
        drift_round: int = 10000,
        horizon: int = 20000,
        seed: int = 42
    ):
        self.rng = np.random.default_rng(seed)
        self.p_new_arrival = p_new_arrival
        self.drift_mode = drift_mode
        self.drift_round = drift_round
        self.horizon = horizon

        # Create drifted archetypes by shifting category affinities cyclically
        self.drifted_archetypes = {}
        for p in PERSONA_ARCHETYPES:
            d_p = dict(p)
            affs = p["category_affinities"]
            cats = list(CATEGORIES)
            # Shift categories by 2
            shifted_affs = {cats[(i + 2) % len(cats)]: affs.get(cats[i], 0.05) for i in range(len(cats))}
            d_p["category_affinities"] = shifted_affs
            d_p["budget_mean"] = p["budget_mean"] * 1.3  # 30% budget shift
            self.drifted_archetypes[p["id"]] = d_p

        # Initialize user pool
        self.users: Dict[int, UserInstance] = {}
        self.next_user_id = 0

        for _ in range(n_initial_users):
            self._create_new_user()

    def _create_new_user(self) -> UserInstance:
        uid = self.next_user_id
        self.next_user_id += 1
        persona = self.rng.choice(PERSONA_ARCHETYPES)
        user = UserInstance(uid, persona, self.rng)
        self.users[uid] = user
        return user

    def sample_active_user(self, current_round: int) -> UserInstance:
        """Samples an arriving user (either a returning user or a new arrival)."""
        is_new = (self.rng.random() < self.p_new_arrival) or (len(self.users) == 0)
        if is_new:
            user = self._create_new_user()
        else:
            # Sample existing user (power law activity)
            user_ids = list(self.users.keys())
            uid = self.rng.choice(user_ids)
            user = self.users[uid]

        # Apply drift if scenario requires
        if self.drift_mode in ("abrupt", "abrupt_drift"):
            if current_round >= self.drift_round:
                user.apply_drift(1.0, self.drifted_archetypes[user.persona_id])
        elif self.drift_mode in ("gradual", "gradual_drift"):
            # Drift gradually between round 5000 and 15000
            start_drift = self.drift_round // 2
            end_drift = self.drift_round + (self.drift_round // 2)
            if current_round >= start_drift:
                ratio = min(1.0, (current_round - start_drift) / max(1.0, end_drift - start_drift))
                user.apply_drift(ratio, self.drifted_archetypes[user.persona_id])

        user.total_sessions += 1
        user.last_round_seen = current_round
        return user

