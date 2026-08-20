"""JSON API routes for AJAX and external integrations."""
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from services.history_service import HistoryService

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")

@api_bp.route("/experiments")
@login_required
def api_experiments():
    exps = HistoryService.get_user_experiments(current_user.id, limit=100)
    return jsonify([{
        "id": e.id,
        "name": e.experiment_name,
        "status": e.status,
        "problem": e.detected_problem,
        "best_model": e.best_model_name,
        "metric": e.best_metric_name,
        "score": e.best_metric_value,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    } for e in exps])

@api_bp.route("/quota")
@login_required
def api_quota():
    from services.subscription_service import SubscriptionService
    sub_svc = SubscriptionService(current_app.config.get("PLANS", {}))
    return jsonify(sub_svc.remaining_quota(current_user))