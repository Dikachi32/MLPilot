"""Subscription and demo payment routes."""
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from services.subscription_service import SubscriptionService
from services.payment_service import DemoPaymentService
from services.usage_service import UsageService

sub_bp = Blueprint("subscription", __name__, url_prefix="/subscription")

@sub_bp.route("/plans")
@login_required
def plans():
    cfg = current_app.config.get("PLANS", {})
    return render_template("subscription/plans.html", plans=cfg, user=current_user)

@sub_bp.route("/checkout", methods=["POST"])
@login_required
def checkout():
    plan = request.form.get("plan")
    billing = request.form.get("billing", "monthly")

    if plan not in current_app.config.get("PLANS", {}):
        flash("Invalid plan selected.", "error")
        return redirect(url_for("subscription.plans"))

    session = DemoPaymentService.create_checkout(plan, billing, current_user.email)
    return render_template("subscription/checkout.html", session=session, plan=plan, billing=billing)

@sub_bp.route("/process", methods=["POST"])
@login_required
def process():
    session_id = request.form.get("session_id")
    plan = request.form.get("plan")
    billing = request.form.get("billing", "monthly")

    if not DemoPaymentService.verify_session(session_id):
        flash("Invalid checkout session.", "error")
        return redirect(url_for("subscription.plans"))

    result = DemoPaymentService.process_payment(session_id, plan, billing)

    # Activate plan
    duration = 7 if billing == "weekly" else 30
    sub_svc = SubscriptionService(current_app.config.get("PLANS", {}))
    sub_svc.activate_plan(current_user, plan, duration_days=duration)

    flash(f"Demo payment successful! Your {plan} plan is now active.", "success")
    return redirect(url_for("main.dashboard"))