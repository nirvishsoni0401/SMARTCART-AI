"""
SmartCart AI — Calibrated Simulator v2: Simulation Environment
High-performance vectorized simulation environment for contextual bandits.
Implements:
  - 262 active catalog products (with 12 cold-start items activated at round 10,000).
  - Slate presentation (K = 10) with position examination bias gamma_k = 1 / log2(k + 2).
  - Latent utility and conditional funnel calibrated to Retailrocket empirical anchors:
      P(view -> cart) ~ 2.6%
      P(cart -> tx) ~ 32.4%
      Random slate CTR ~ 3-6%
      Oracle slate CTR ~ 20-35%
  - 3 difficulty levels: Easy (sigma=0.15), Medium (sigma=0.40), Hard (sigma=0.80).
  - Analytical expected reward and exact regret calculation (Oracle regret == 0, Random regret > 0).
  - Vectorized expected value computation for high-speed 20,000-round benchmarks.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import math
import numpy as np
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Product
from app.services.feature_engineering import (
    CATEGORIES,
    CATEGORY_INDEX,
    MAX_PRICE_SCALE,
    UserContext,
)
from experiments.sim_v2.user_model import UserManager, UserInstance

# Monotonic examination bias gamma_k = 1 / log2(k + 2) for k in 0..9
POSITION_BIAS = np.array([1.0 / math.log2(k + 2) for k in range(10)], dtype=np.float64)

# Central reward specification from app/config.py
REWARD_CONFIG = {
    "purchase": 10.0,
    "cart_add": 5.0,
    "wishlist": 3.0,
    "click": 2.0,
    "dislike": -5.0,
    "cart_remove": -2.0,
    "return": -8.0,
    "skip": 0.0,
}

# Funnel conditional base probabilities (Retailrocket anchors)
BASE_P_CART_GIVEN_CLICK = 0.026
BASE_P_TX_GIVEN_CART = 0.324
P_RETURN_GIVEN_TX = 0.035
P_REMOVE_GIVEN_NO_TX = 0.050
P_WISHLIST_GIVEN_NO_CART = 0.060
P_DISLIKE_GIVEN_NO_CART = 0.040

# Difficulty noise scale (sigma of Gaussian noise in latent utility)
DIFFICULTY_SIGMA = {
    "easy": 0.15,
    "medium": 0.40,
    "hard": 0.80,
}


class SlateFeedback:
    """Feedback returned to the simulation harness after a slate presentation."""

    def __init__(
        self,
        round_idx: int,
        user: UserInstance,
        user_context: UserContext,
        slate_pids: List[int],
        events: List[Dict[str, Any]],
        realized_reward: float,
        expected_policy_reward: float,
        expected_oracle_reward: float,
        expected_regret: float,
        oracle_slate: List[int],
        clicked_pids: List[int],
        cart_pids: List[int],
        purchased_pids: List[int],
        disliked_pids: List[int],
        feature_matrix: np.ndarray,
    ):
        self.round_idx = round_idx
        self.user = user
        self.user_context = user_context
        self.slate_pids = slate_pids
        self.events = events
        self.realized_reward = realized_reward
        self.expected_policy_reward = expected_policy_reward
        self.expected_oracle_reward = expected_oracle_reward
        self.expected_regret = expected_regret
        self.oracle_slate = oracle_slate
        self.clicked_pids = clicked_pids
        self.cart_pids = cart_pids
        self.purchased_pids = purchased_pids
        self.disliked_pids = disliked_pids
        self.feature_matrix = feature_matrix  # (K, 36) features of displayed items


class EnvironmentV2:
    """
    Calibrated simulation environment v2.
    Loads the 262 active products and provides high-speed vectorized slate interaction steps.
    """

    def __init__(
        self,
        difficulty: str = "medium",
        drift_mode: str = "none",  # "none", "abrupt", "gradual"
        drift_round: int = 10000,
        horizon: int = 20000,
        seed: int = 42,
        db_session: Optional[Session] = None
    ):
        self.difficulty = difficulty.lower()
        self.sigma = DIFFICULTY_SIGMA.get(self.difficulty, 0.40)
        self.drift_mode = drift_mode
        self.drift_round = drift_round
        self.horizon = horizon
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Initialize User Manager
        self.user_manager = UserManager(
            n_initial_users=150,
            p_new_arrival=0.25,
            drift_mode=drift_mode,
            drift_round=drift_round,
            horizon=horizon,
            seed=seed
        )

        # Load active products from database
        close_db = False
        if db_session is None:
            db_session = SessionLocal()
            close_db = True

        try:
            db_products = db_session.query(Product).filter(Product.is_active == True).order_by(Product.id.asc()).all()
            self.products = [
                {
                    "id": p.id,
                    "name": p.name,
                    "category": p.category,
                    "brand": p.brand or "Unbranded",
                    "price": float(p.price),
                    "rating": float(p.rating or 4.0),
                    "discount_pct": float(p.discount_pct or 0.0),
                    "popularity_score": float(p.popularity_score or 50.0),
                }
                for p in db_products
            ]
        finally:
            if close_db:
                db_session.close()

        self.n_products = len(self.products)
        self.pid_to_index = {p["id"]: i for i, p in enumerate(self.products)}
        self.index_to_pid = np.array([p["id"] for p in self.products], dtype=np.int64)

        # Vectorized product attributes
        self.prod_cats = np.array([CATEGORY_INDEX.get(p["category"], 0) for p in self.products], dtype=np.int32)
        self.prod_prices = np.array([p["price"] for p in self.products], dtype=np.float64)
        self.prod_ratings = np.array([p["rating"] / 5.0 for p in self.products], dtype=np.float64)
        self.prod_discounts = np.array([p["discount_pct"] / 50.0 for p in self.products], dtype=np.float64)
        self.prod_brands = [p["brand"] for p in self.products]

        # Select 12 cold-start products (2 from each category with highest IDs)
        # These 12 start inactive and become available at round 10,000
        cat_items: Dict[str, List[int]] = {c: [] for c in CATEGORIES}
        for p in self.products:
            if p["category"] in cat_items:
                cat_items[p["category"]].append(p["id"])

        cold_start_pids = set()
        for cat, pids in cat_items.items():
            sorted_pids = sorted(pids)
            if len(sorted_pids) >= 2:
                cold_start_pids.update(sorted_pids[-2:])
        self.cold_start_pids = cold_start_pids
        self.initial_pids = [p["id"] for p in self.products if p["id"] not in self.cold_start_pids]
        self.all_pids = [p["id"] for p in self.products]

        # Boolean mask of available items
        self.initial_mask = np.array([p["id"] not in self.cold_start_pids for p in self.products], dtype=bool)

        # Precompute static feature components for all 262 products
        self._precompute_static_features()

        # Calibration weights for latent logit utility
        self.w_cat = 2.2
        self.w_brand = 0.8
        self.w_rating = 0.6
        self.w_price = 1.0
        self.w_disc = 0.5
        self.theta_click = 5.72

    def _precompute_static_features(self):
        """Precompute static feature matrix (N, 36) for columns 12..30 and 33..35."""
        self.static_features = np.zeros((self.n_products, 36), dtype=np.float64)
        for i, p in enumerate(self.products):
            # 12 - 17: Product category one-hot
            cat_idx = CATEGORY_INDEX.get(p["category"])
            if cat_idx is not None:
                self.static_features[i, 12 + cat_idx] = 1.0

            # 18: Product price log
            self.static_features[i, 18] = math.log(1.0 + p["price"]) / MAX_PRICE_SCALE

            # 19: Product rating norm
            self.static_features[i, 19] = max(0.0, min(1.0, (p["rating"] - 1.0) / 4.0))

            # 20: Discount pct
            self.static_features[i, 20] = max(0.0, min(1.0, p["discount_pct"] / 100.0))

            # 21: Popularity score
            self.static_features[i, 21] = max(0.0, min(1.0, p["popularity_score"] / 100.0))

            # 22 - 29: Brand hash bucket
            bucket = abs(hash(p["brand"])) % 8
            self.static_features[i, 22 + bucket] = 1.0

            # 30: New product flag
            self.static_features[i, 30] = 1.0 if p["id"] > 550 else 0.0

            # 33: High rating flag (>= 4.5)
            self.static_features[i, 33] = 1.0 if p["rating"] >= 4.5 else 0.0

            # 34: High discount flag (>= 30%)
            self.static_features[i, 34] = 1.0 if p["discount_pct"] >= 30.0 else 0.0

            # 35: Intercept
            self.static_features[i, 35] = 1.0

    def get_available_product_ids(self, current_round: int) -> List[int]:
        """Returns the list of product IDs available to policies at current_round."""
        if current_round < self.drift_round:
            return self.initial_pids
        return self.all_pids

    def build_user_context(self, user: UserInstance) -> UserContext:
        """
        Builds UserContext efficiently from user state without heavy database or string parsing.
        """
        # Category affinities normalized
        cat_aff = {cat: user.category_affinities.get(cat, 0.0) for cat in CATEGORIES}
        # Brand affinities
        brand_aff = {b: 1.0 for b in user.preferred_brands}
        # Cart categories
        cart_cats = {}
        for pid in user.cart:
            p_cat = self.products[self.pid_to_index[pid]]["category"]
            cart_cats[p_cat] = cart_cats.get(p_cat, 0) + 1

        avg_price = user.budget if user.budget > 0 else 1800.0

        return UserContext(
            user_id=user.user_id,
            session_id=f"sim_sess_{user.user_id}_{user.total_sessions}",
            session_budget=user.budget,
            category_affinities=cat_aff,
            brand_affinities=brand_aff,
            avg_price=avg_price,
            total_interactions=len(user.history),
            cart_categories=cart_cats
        )

    def build_feature_matrix_for_user(self, user: UserInstance, user_context: UserContext) -> np.ndarray:
        """
        Constructs the (N, 36) feature matrix for all 262 products given arriving user context.
        Vectorized for maximum simulation throughput.
        """
        phi_matrix = self.static_features.copy()

        # Dynamic Columns:
        # 0 - 5: Category affinity
        for c_idx, cat in enumerate(CATEGORIES):
            phi_matrix[:, c_idx] = user_context.category_affinities.get(cat, 0.0)

        # 6: Brand affinity
        for i, b in enumerate(self.prod_brands):
            phi_matrix[i, 6] = user_context.brand_affinities.get(b, 0.0)

        # 7: Avg historical price level
        ref_price = user_context.avg_price if user_context.avg_price is not None else 1800.0
        phi_matrix[:, 7] = math.log(1.0 + ref_price) / MAX_PRICE_SCALE

        # 8: Budget log
        budget = user_context.session_budget if user_context.session_budget is not None else ref_price
        budget_log = math.log(1.0 + budget)
        phi_matrix[:, 8] = budget_log / MAX_PRICE_SCALE

        # 9: Price fit score & 10: in-budget
        p_logs = np.log1p(self.prod_prices)
        price_diffs = np.abs(p_logs - budget_log)
        phi_matrix[:, 9] = np.maximum(0.0, 1.0 - (price_diffs / 2.0))
        phi_matrix[:, 10] = (self.prod_prices <= budget).astype(np.float64)

        # 31: User activity level
        phi_matrix[:, 31] = min(1.0, math.log(1.0 + user_context.total_interactions) / math.log(50.0))

        # 32: Cart intent match
        total_cart = sum(user_context.cart_categories.values())
        if total_cart > 0:
            for i, c_idx in enumerate(self.prod_cats):
                cat_name = CATEGORIES[c_idx]
                cat_cart = user_context.cart_categories.get(cat_name, 0)
                phi_matrix[i, 32] = cat_cart / total_cart
        else:
            phi_matrix[:, 32] = 0.0

        return phi_matrix

    def compute_all_latent_click_probabilities(
        self,
        user: UserInstance,
        round_idx: int,
        with_noise: bool = True
    ) -> np.ndarray:
        """
        Vectorized computation of P(click | examined) for all 262 products.
        """
        # Category affinity per product
        cat_aff_vec = np.array([user.category_affinities.get(c, 0.05) for c in CATEGORIES], dtype=np.float64)
        cat_affs = cat_aff_vec[self.prod_cats]

        # Brand match
        if user.preferred_brands:
            brand_matches = np.array([1.0 if b in user.preferred_brands else 0.0 for b in self.prod_brands], dtype=np.float64)
        else:
            brand_matches = np.zeros(self.n_products, dtype=np.float64)

        # Price penalty
        budget = max(1.0, user.budget)
        excess = np.maximum(0.0, self.prod_prices - budget) / budget
        price_penalties = np.minimum(2.5, excess) * user.price_sensitivity

        # Discount incentive
        disc_incentives = np.minimum(1.0, self.prod_discounts) * user.discount_sensitivity

        # Repeat bonus
        repeat_bonuses = np.zeros(self.n_products, dtype=np.float64)
        viewed_pids = user.get_viewed_product_ids()
        if viewed_pids:
            recency_decay = 0.75 * math.exp(-0.0005 * max(0, round_idx - user.last_round_seen))
            for vpid in viewed_pids:
                if vpid in self.pid_to_index:
                    repeat_bonuses[self.pid_to_index[vpid]] = recency_decay

        # Latent logit
        logits = (
            self.w_cat * cat_affs
            + self.w_brand * brand_matches
            + self.w_rating * self.prod_ratings
            - self.w_price * price_penalties
            + self.w_disc * disc_incentives
            + repeat_bonuses
            - self.theta_click
        )

        if with_noise:
            logits += self.rng.normal(0.0, self.sigma, size=self.n_products)

        # Sigmoid
        p_clicks = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        return p_clicks

    def compute_all_expected_item_values(
        self,
        user: UserInstance,
        round_idx: int
    ) -> np.ndarray:
        """
        Vectorized computation of exact noiseless expected reward V(u, i) for all 262 products.
        """
        p_clicks = self.compute_all_latent_click_probabilities(user, round_idx, with_noise=False)

        cat_aff_vec = np.array([user.category_affinities.get(c, 0.05) for c in CATEGORIES], dtype=np.float64)
        cat_affs = cat_aff_vec[self.prod_cats]

        budget = max(1.0, user.budget)
        excess = np.maximum(0.0, self.prod_prices - budget) / budget
        in_budget = (self.prod_prices <= budget).astype(np.float64)

        cart_mods = np.clip(0.5 + 0.5 * cat_affs + 0.3 * in_budget - 0.3 * excess, 0.5, 2.0)
        p_carts = BASE_P_CART_GIVEN_CLICK * cart_mods

        tx_mods = np.clip(0.6 + 0.4 * self.prod_ratings + 0.2 * self.prod_discounts, 0.6, 1.5)
        p_txs = BASE_P_TX_GIVEN_CART * tx_mods

        # Expected reward values
        e_tx = REWARD_CONFIG["purchase"] + P_RETURN_GIVEN_TX * REWARD_CONFIG["return"]  # 10 - 0.28 = 9.72
        e_no_tx = P_REMOVE_GIVEN_NO_TX * REWARD_CONFIG["cart_remove"]                   # -0.10
        e_carts = REWARD_CONFIG["cart_add"] + p_txs * e_tx + (1.0 - p_txs) * e_no_tx   # 5 + p_tx*9.72 - (1-p_tx)*0.1
        e_no_cart = (
            P_WISHLIST_GIVEN_NO_CART * REWARD_CONFIG["wishlist"]
            + P_DISLIKE_GIVEN_NO_CART * REWARD_CONFIG["dislike"]
        )  # 0.18 - 0.20 = -0.02
        e_clicks = REWARD_CONFIG["click"] + p_carts * e_carts + (1.0 - p_carts) * e_no_cart

        expected_values = p_clicks * e_clicks
        return expected_values

    def get_oracle_slate(
        self,
        user: UserInstance,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        expected_values: Optional[np.ndarray] = None
    ) -> Tuple[List[int], float]:
        """
        Selects the optimal top-k items that maximize expected slate reward.
        Returns: (oracle_slate_pids, expected_oracle_reward)
        """
        if expected_values is None:
            expected_values = self.compute_all_expected_item_values(user, round_idx)

        # Get indices of available products
        avail_indices = np.array([self.pid_to_index[pid] for pid in available_pids], dtype=np.int32)
        avail_values = expected_values[avail_indices]

        # Top k indices sorted descending
        top_k_sub_idx = np.argsort(-avail_values)[:k]
        oracle_slate = [available_pids[i] for i in top_k_sub_idx]
        sorted_vals = avail_values[top_k_sub_idx]

        k_eval = min(len(sorted_vals), len(POSITION_BIAS))
        expected_reward = float(np.sum(POSITION_BIAS[:k_eval] * sorted_vals[:k_eval]))
        return oracle_slate, expected_reward

    def step(
        self,
        round_idx: int,
        policy_slate: List[int],
        user: Optional[UserInstance] = None,
        cached_features: Optional[np.ndarray] = None,
        cached_expected_values: Optional[np.ndarray] = None
    ) -> SlateFeedback:
        """
        Executes one high-speed simulation interaction step.
        """
        if user is None:
            user = self.user_manager.sample_active_user(round_idx)

        user_context = self.build_user_context(user)
        available_pids = self.get_available_product_ids(round_idx)

        # Compute or reuse expected values
        if cached_expected_values is None:
            all_expected_values = self.compute_all_expected_item_values(user, round_idx)
        else:
            all_expected_values = cached_expected_values

        oracle_slate, e_r_oracle = self.get_oracle_slate(
            user, available_pids, round_idx, k=10, expected_values=all_expected_values
        )

        # Policy slate expected reward
        policy_indices = [self.pid_to_index[pid] for pid in policy_slate[:10]]
        policy_vals = all_expected_values[policy_indices]
        e_r_policy = float(np.sum(POSITION_BIAS[:len(policy_vals)] * policy_vals))

        # Expected regret is strictly >= 0 (identically 0 for Oracle)
        expected_regret = max(0.0, e_r_oracle - e_r_policy)

        # Feature matrix for the displayed slate
        if cached_features is not None:
            feature_matrix = cached_features[policy_indices]
        else:
            full_feature_mat = self.build_feature_matrix_for_user(user, user_context)
            feature_matrix = full_feature_mat[policy_indices]

        # Latent click probabilities with noise for the displayed slate
        displayed_p_clicks = self.compute_all_latent_click_probabilities(user, round_idx, with_noise=True)[policy_indices]

        # Simulate realization through the conditional funnel
        events = []
        realized_reward = 0.0
        clicked_pids = []
        cart_pids = []
        purchased_pids = []
        disliked_pids = []

        budget = max(1.0, user.budget)

        for pos, pid in enumerate(policy_slate[:10]):
            gamma_k = POSITION_BIAS[pos]

            # 1. Position examination
            if self.rng.random() >= gamma_k:
                continue

            # 2. Click decision
            if self.rng.random() >= displayed_p_clicks[pos]:
                continue

            # Click occurred!
            p_idx = policy_indices[pos]
            p = self.products[p_idx]
            clicked_pids.append(pid)
            realized_reward += REWARD_CONFIG["click"]
            user.history.append((pid, round_idx, "click"))
            events.append({"pid": pid, "event_type": "click", "reward": REWARD_CONFIG["click"], "position": pos})

            # 3. Add-to-cart decision
            cat_aff = user.category_affinities.get(p["category"], 0.05)
            excess = max(0.0, (p["price"] - budget) / budget)
            in_budget = 1.0 if p["price"] <= budget else 0.0
            cart_mod = np.clip(0.5 + 0.5 * cat_aff + 0.3 * in_budget - 0.3 * excess, 0.5, 2.0)
            p_cart = BASE_P_CART_GIVEN_CLICK * cart_mod

            if self.rng.random() < p_cart:
                # Add to cart!
                cart_pids.append(pid)
                user.cart.add(pid)
                realized_reward += REWARD_CONFIG["cart_add"]
                user.history.append((pid, round_idx, "cart_add"))
                events.append({"pid": pid, "event_type": "cart_add", "reward": REWARD_CONFIG["cart_add"], "position": pos})

                # 4. Transaction decision
                tx_mod = np.clip(0.6 + 0.4 * self.prod_ratings[p_idx] + 0.2 * self.prod_discounts[p_idx], 0.6, 1.5)
                p_tx = BASE_P_TX_GIVEN_CART * tx_mod

                if self.rng.random() < p_tx:
                    # Purchase!
                    purchased_pids.append(pid)
                    user.purchased.add(pid)
                    user.cart.discard(pid)
                    realized_reward += REWARD_CONFIG["purchase"]
                    user.history.append((pid, round_idx, "purchase"))
                    events.append({"pid": pid, "event_type": "purchase", "reward": REWARD_CONFIG["purchase"], "position": pos})

                    # Delayed return
                    if self.rng.random() < P_RETURN_GIVEN_TX:
                        realized_reward += REWARD_CONFIG["return"]
                        user.history.append((pid, round_idx, "return"))
                        events.append({"pid": pid, "event_type": "return", "reward": REWARD_CONFIG["return"], "position": pos})
                else:
                    # Cart removal
                    if self.rng.random() < P_REMOVE_GIVEN_NO_TX:
                        user.cart.discard(pid)
                        realized_reward += REWARD_CONFIG["cart_remove"]
                        user.history.append((pid, round_idx, "cart_remove"))
                        events.append({"pid": pid, "event_type": "cart_remove", "reward": REWARD_CONFIG["cart_remove"], "position": pos})
            else:
                # No cart: wishlist or dislike
                r_other = self.rng.random()
                if r_other < P_WISHLIST_GIVEN_NO_CART:
                    user.wishlist.add(pid)
                    realized_reward += REWARD_CONFIG["wishlist"]
                    user.history.append((pid, round_idx, "wishlist"))
                    events.append({"pid": pid, "event_type": "wishlist", "reward": REWARD_CONFIG["wishlist"], "position": pos})
                elif r_other < (P_WISHLIST_GIVEN_NO_CART + P_DISLIKE_GIVEN_NO_CART):
                    disliked_pids.append(pid)
                    realized_reward += REWARD_CONFIG["dislike"]
                    user.history.append((pid, round_idx, "dislike"))
                    events.append({"pid": pid, "event_type": "dislike", "reward": REWARD_CONFIG["dislike"], "position": pos})

        return SlateFeedback(
            round_idx=round_idx,
            user=user,
            user_context=user_context,
            slate_pids=policy_slate[:10],
            events=events,
            realized_reward=realized_reward,
            expected_policy_reward=e_r_policy,
            expected_oracle_reward=e_r_oracle,
            expected_regret=expected_regret,
            oracle_slate=oracle_slate,
            clicked_pids=clicked_pids,
            cart_pids=cart_pids,
            purchased_pids=purchased_pids,
            disliked_pids=disliked_pids,
            feature_matrix=feature_matrix
        )
