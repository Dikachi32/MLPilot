"""Authentication routes: register, login, logout, profile."""
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from models.user import User, db

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        full_name = request.form.get("full_name", "").strip()

        if not email or not username or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("auth.register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return redirect(url_for("auth.register"))
        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "error")
            return redirect(url_for("auth.register"))

        user = User(email=email, username=username, full_name=full_name)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # Auto-start free trial
        from services.subscription_service import SubscriptionService
        from flask import current_app
        sub_svc = SubscriptionService(current_app.config.get('PLANS', {}))
        sub_svc.start_trial(user)

        login_user(user)
        flash("Welcome to MLPilot! Your free trial has started.", "success")
        return redirect(url_for("main.dashboard"))
    return render_template("auth/register.html")

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter((User.email == identifier.lower()) | (User.username == identifier)).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("main.dashboard"))
        flash("Invalid credentials.", "error")
    return render_template("auth/login.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))

@auth_bp.route("/profile")
@login_required
def profile():
    from services.subscription_service import SubscriptionService
    sub_svc = SubscriptionService({})
    quota = sub_svc.remaining_quota(current_user)
    return render_template("auth/profile.html", user=current_user, quota=quota)