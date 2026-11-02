import datetime
from typing import Optional, Dict, Any
from app.database import bandit_models_collection
from app.recommendation.linucb import LinUCBAgent

ACTIVE_MODEL_VERSION = "linucb_v1"


def save_bandit_model(agent: LinUCBAgent, version: str = ACTIVE_MODEL_VERSION, metadata: Optional[Dict[str, Any]] = None):
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    doc = {
        "version": version,
        "algorithm": "Contextual_LinUCB",
        "d": agent.d,
        "alpha": agent.alpha,
        "l2_reg": agent.l2_reg,
        "arms": agent.to_dict()["arms"],
        "metadata": metadata or {},
        "updated_at": now_iso
    }
    bandit_models_collection.update_one(
        {"version": version},
        {"$set": doc, "$setOnInsert": {"created_at": now_iso}},
        upsert=True
    )


def load_bandit_model(version: str = ACTIVE_MODEL_VERSION) -> LinUCBAgent:
    doc = bandit_models_collection.find_one({"version": version})
    if not doc:
        agent = LinUCBAgent()
        save_bandit_model(agent, version=version)
        return agent
    
    return LinUCBAgent.from_dict({
        "d": doc.get("d", 8),
        "alpha": doc.get("alpha", 0.8),
        "l2_reg": doc.get("l2_reg", 1.0),
        "arms": doc.get("arms", {})
    })

