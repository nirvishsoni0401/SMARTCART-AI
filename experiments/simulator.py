import math
import random
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class CustomerPersona:
    """Represents a simulated shopper with hidden ground-truth preferences."""
    def __init__(
        self,
        name: str,
        preferred_categories: Dict[str, float],
        preferred_brands: Dict[str, float],
        min_budget: float,
        max_budget: float,
        price_sensitivity: float,
        quality_sensitivity: float,
        deal_sensitivity: float
    ):
        self.name = name
        self.preferred_categories = preferred_categories
        self.preferred_brands = preferred_brands
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.price_sensitivity = price_sensitivity
        self.quality_sensitivity = quality_sensitivity
        self.deal_sensitivity = deal_sensitivity

    def compute_hidden_utility(self, product: Any) -> float:
        """Ground-truth utility in [-1, 1]."""
        cat = getattr(product, "category", "")
        brand = getattr(product, "brand", "")
        price = getattr(product, "price", 1000.0)
        rating = getattr(product, "rating", 4.0)
        discount = getattr(product, "discount_pct", 10.0)

        # 1. Category taste component
        cat_score = self.preferred_categories.get(cat, 0.0) * 1.5

        # 2. Brand taste component
        brand_score = self.preferred_brands.get(brand, 0.0) * 0.8

        # 3. Budget fit
        if price > self.max_budget:
            budget_penalty = -self.price_sensitivity * min(2.0, (price - self.max_budget) / self.max_budget)
        elif price < self.min_budget:
            budget_penalty = -0.2 * self.price_sensitivity
        else:
            budget_penalty = 0.4

        # 4. Rating / quality preference
        rating_score = self.quality_sensitivity * ((rating - 3.5) / 1.5)

        # 5. Deal / discount preference
        deal_score = self.deal_sensitivity * (discount / 60.0)

        raw_utility = cat_score + brand_score + budget_penalty + rating_score + deal_score - 0.3
        # Sigmoidal clamp to [-1, 1]
        return 2.0 / (1.0 + math.exp(-raw_utility)) - 1.0

def create_default_personas() -> List[CustomerPersona]:
    """Create 8 diverse simulated customer personas."""
    return [
        CustomerPersona(
            name="TechEnthusiast",
            preferred_categories={"Electronics": 1.0, "Sports": 0.2},
            preferred_brands={"Sony": 0.9, "OnePlus": 0.85, "Samsung": 0.8, "boAt": 0.7},
            min_budget=1500.0,
            max_budget=45000.0,
            price_sensitivity=0.3,
            quality_sensitivity=0.9,
            deal_sensitivity=0.4
        ),
        CustomerPersona(
            name="BudgetStudent",
            preferred_categories={"Books": 0.9, "Fashion": 0.7, "Electronics": 0.5},
            preferred_brands={"boAt": 0.8, "Peter England": 0.7, "Jaico": 0.8, "Zebronics": 0.7},
            min_budget=299.0,
            max_budget=1999.0,
            price_sensitivity=1.2,
            quality_sensitivity=0.4,
            deal_sensitivity=1.0
        ),
        CustomerPersona(
            name="LuxuryShopper",
            preferred_categories={"Fashion": 1.0, "Beauty": 0.8, "Electronics": 0.7},
            preferred_brands={"Titan": 0.9, "Raymond": 0.85, "Sony": 0.8, "Forest Essentials": 0.9},
            min_budget=3000.0,
            max_budget=80000.0,
            price_sensitivity=0.1,
            quality_sensitivity=1.1,
            deal_sensitivity=0.2
        ),
        CustomerPersona(
            name="Bookworm",
            preferred_categories={"Books": 1.0},
            preferred_brands={"O'Reilly": 0.9, "Penguin India": 0.9, "HarperCollins": 0.85, "Packt": 0.8},
            min_budget=350.0,
            max_budget=2500.0,
            price_sensitivity=0.5,
            quality_sensitivity=0.9,
            deal_sensitivity=0.5
        ),
        CustomerPersona(
            name="FitnessBuff",
            preferred_categories={"Sports": 1.0, "Beauty": 0.6},
            preferred_brands={"HRX": 0.9, "Cultsport": 0.85, "Nivia": 0.8, "Boldfit": 0.8, "mCaffeine": 0.7},
            min_budget=500.0,
            max_budget=8000.0,
            price_sensitivity=0.5,
            quality_sensitivity=0.8,
            deal_sensitivity=0.6
        ),
        CustomerPersona(
            name="HomeDecorator",
            preferred_categories={"Home & Kitchen": 1.0, "Beauty": 0.3},
            preferred_brands={"Prestige": 0.9, "Hawkins": 0.85, "Milton": 0.8, "Borosil": 0.85, "Bajaj": 0.75},
            min_budget=800.0,
            max_budget=12000.0,
            price_sensitivity=0.6,
            quality_sensitivity=0.75,
            deal_sensitivity=0.7
        ),
        CustomerPersona(
            name="BargainHunter",
            preferred_categories={"Fashion": 0.6, "Home & Kitchen": 0.6, "Electronics": 0.6, "Beauty": 0.6},
            preferred_brands={},
            min_budget=200.0,
            max_budget=3500.0,
            price_sensitivity=1.3,
            quality_sensitivity=0.3,
            deal_sensitivity=1.4
        ),
        CustomerPersona(
            name="TrendFollower",
            preferred_categories={"Fashion": 0.9, "Beauty": 0.85},
            preferred_brands={"Mamaearth": 0.8, "Plum": 0.8, "Dot & Key": 0.85, "Biba": 0.8},
            min_budget=800.0,
            max_budget=6000.0,
            price_sensitivity=0.5,
            quality_sensitivity=0.7,
            deal_sensitivity=0.7
        )
    ]

class ShoppingSimulator:
    """
    Simulated e-commerce shopping environment with probabilistic response model
    and Oracle policy for exact regret computation.
    """

    def __init__(self, catalog_products: List[Any], seed: int = 42):
        self.catalog = catalog_products
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.personas = create_default_personas()
        self.drift_active = False

    def trigger_preference_drift(self):
        """Simulate preference drift midway: personas shift taste and budgets."""
        self.drift_active = True
        for p in self.personas:
            if p.name == "TechEnthusiast":
                # Shifts from Electronics to Home & Kitchen
                p.preferred_categories = {"Home & Kitchen": 1.0, "Books": 0.4}
                p.max_budget = 15000.0
            elif p.name == "BudgetStudent":
                # Starts earning, shifts to Electronics and Sports
                p.preferred_categories = {"Electronics": 0.9, "Sports": 0.8}
                p.max_budget = 12000.0
                p.price_sensitivity = 0.6

    def sample_customer_session(self) -> Tuple[CustomerPersona, Dict[str, Any]]:
        persona = self.rng.choice(self.personas)
        session_cat = self.rng.choice(list(persona.preferred_categories.keys())) if self.rng.random() < 0.65 else None
        session_budget = self.rng.uniform(persona.min_budget, persona.max_budget) if self.rng.random() < 0.5 else None

        context = {
            "session_id": f"sim_{self.rng.randint(1000, 99999)}",
            "persona_name": persona.name,
            "session_category": session_cat,
            "session_budget": session_budget
        }
        return persona, context

    def simulate_user_response(
        self,
        persona: CustomerPersona,
        product: Any
    ) -> Tuple[str, float, float]:
        """
        Probabilistic response model:
        Returns (event_type, reward, ground_truth_utility)
        Event types: click, cart_add, purchase, dislike, return, skip
        """
        u = persona.compute_hidden_utility(product)

        # Baseline click probability based on utility
        p_click = 1.0 / (1.0 + math.exp(-2.8 * u))
        p_click = min(0.92, max(0.04, p_click))

        roll = self.rng.random()

        if roll < p_click:
            # Clicked! Now check cart addition
            p_cart = min(0.85, max(0.1, 0.45 * (u + 0.5)))
            if self.rng.random() < p_cart:
                # Cart added! Check purchase
                p_buy = min(0.85, max(0.15, 0.55 * (u + 0.5)))
                if self.rng.random() < p_buy:
                    # Purchased! Check potential return
                    p_return = 0.08 * (1.0 - getattr(product, "rating", 4.0) / 5.0)
                    if self.rng.random() < p_return:
                        return "return", -8.0, u
                    return "purchase", 10.0, u
                return "cart_add", 5.0, u
            return "click", 2.0, u
        else:
            # Unclicked / skipped. Check if disliked
            p_dislike = 1.0 / (1.0 + math.exp(3.2 * u))  # high when utility is negative
            p_dislike = min(0.35, max(0.01, p_dislike * 0.4))
            if self.rng.random() < p_dislike:
                return "dislike", -5.0, u
            return "skip", 0.0, u

    def compute_oracle_best_reward(
        self,
        persona: CustomerPersona,
        candidate_products: List[Any]
    ) -> Tuple[float, Any]:
        """
        Oracle Policy: evaluates all candidates using persona's ground truth utility
        and returns the expected maximum reward possible.
        """
        best_reward = -float("inf")
        best_prod = None

        for prod in candidate_products:
            u = persona.compute_hidden_utility(prod)
            # Expected reward under ground truth utility
            p_click = 1.0 / (1.0 + math.exp(-2.8 * u))
            p_cart = p_click * min(0.85, max(0.1, 0.45 * (u + 0.5)))
            p_buy = p_cart * min(0.85, max(0.15, 0.55 * (u + 0.5)))
            
            exp_r = p_buy * 10.0 + (p_cart - p_buy) * 5.0 + (p_click - p_cart) * 2.0
            if exp_r > best_reward:
                best_reward = exp_r
                best_prod = prod

        return max(0.0, best_reward), best_prod

