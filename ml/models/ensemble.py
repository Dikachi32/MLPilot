"""Ensemble methods builder."""
from sklearn.ensemble import VotingClassifier, VotingRegressor
from typing import Dict, Any, List

def build_voting_ensemble(models: Dict[str, Any], problem_type: str) -> Any:
    """Build a Voting ensemble from trained models."""
    estimators = [(name, m) for name, m in models.items()]
    if not estimators:
        return None
    if problem_type == "classification":
        return VotingClassifier(estimators=estimators, voting="soft")
    else:
        return VotingRegressor(estimators=estimators)