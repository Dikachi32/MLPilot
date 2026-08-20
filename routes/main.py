"""Main application routes: dashboard, upload, AutoML builder, results."""
import os
import uuid
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, send_file, session as flask_session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from models.user import db
from services.usage_service import UsageService
from services.history_service import HistoryService
from services.subscription_service import SubscriptionService
from ml.pipeline import MLPilotPipeline
from ml.detection import analyze_dataset, suggest_target_columns

main_bp = Blueprint("main", __name__)



def convert_to_native(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native(v) for v in obj)
    return obj

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv", "zip"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    return render_template("landing.html")

@main_bp.route("/dashboard")
@login_required
def dashboard():
    experiments = HistoryService.get_user_experiments(current_user.id, limit=10)
    sub_svc = SubscriptionService(current_app.config.get("PLANS", {}))
    quota = sub_svc.remaining_quota(current_user)
    return render_template("dashboard.html", experiments=experiments, quota=quota, user=current_user)

@main_bp.route("/builder", methods=["GET", "POST"])
@login_required
def builder():
    usage = UsageService(current_user)

    if request.method == "POST":
        ok, msg = usage.check_and_increment("dataset_upload")
        if not ok:
            flash(msg, "error")
            return redirect(url_for("main.builder"))

        file = request.files.get("dataset")
        if not file or file.filename == "":
            flash("Please select a file.", "error")
            return redirect(url_for("main.builder"))

        if not allowed_file(file.filename):
            flash("Only CSV and ZIP files are allowed.", "error")
            return redirect(url_for("main.builder"))

        ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex[:8]}.{ext}"
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
        os.makedirs(current_app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(filepath)

        # Detect encoding and read
        from ml.preprocessing import PreprocessingEngine
        try:
            df, df_raw, encoding = read_csv_robust(filepath)
        except Exception as e:
            flash(f"Could not read file: {str(e)}", "error")
            return redirect(url_for("main.builder"))

        # Validate size
        cfg = current_user.get_plan_config(current_app.config)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > cfg.get("max_dataset_size_mb", 10):
            flash(f"Dataset too large ({size_mb:.1f} MB). Plan limit: {cfg['max_dataset_size_mb']} MB.", "error")
            return redirect(url_for("main.builder"))

        analysis = analyze_dataset(df)
        suggested_targets = suggest_target_columns(df, analysis)

        # Store in session for builder flow
        flask_session["current_dataset"] = unique_name
        flask_session["current_analysis"] = analysis

        return render_template("builder.html",
                               dataset_name=file.filename,
                               dataset_path=unique_name,
                               analysis=analysis,
                               suggested_targets=suggested_targets,
                               columns=df.columns.tolist(),
                               encoding=encoding)

    return render_template("builder.html")

@main_bp.route("/train", methods=["POST"])
@login_required
def train():
    usage = UsageService(current_user)
    ok, msg = usage.check_and_increment("train")
    if not ok:
        flash(msg, "error")
        return redirect(url_for("main.builder"))

    dataset_name = flask_session.get("current_dataset")
    if not dataset_name:
        flash("No dataset found. Please upload first.", "error")
        return redirect(url_for("main.builder"))

    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], dataset_name)
    if not os.path.exists(filepath):
        flash("Dataset not found. Please upload again.", "error")
        return redirect(url_for("main.builder"))

    try:
        df, _, _ = read_csv_robust(filepath)
    except Exception as e:
        flash(f"Error reading dataset: {str(e)}", "error")
        return redirect(url_for("main.builder"))

    target_col = request.form.get("target_column") or None
    run_mode = request.form.get("run_mode", "auto")
    optimize = request.form.get("optimize", "false").lower() == "true"

    # Check optimization entitlement
    if optimize:
        cfg = current_user.get_plan_config(current_app.config)
        if not cfg.get("allow_advanced_optimization", False):
            flash("Optimization is not available on your plan. Upgrade to Pro.", "warning")
            optimize = False
        else:
            ok2, msg2 = usage.check_and_increment("optimize")
            if not ok2:
                flash(msg2, "error")
                optimize = False

    # Create experiment record
    exp = HistoryService.create_experiment(
        current_user.id,
        name=f"Experiment {dataset_name[:20]}",
        dataset_name=dataset_name
    )

    # Run pipeline
    pipeline = MLPilotPipeline(artifact_folder=current_app.config["ARTIFACT_FOLDER"])
    result = pipeline.run(df, target_col=target_col, user_plan=current_user.plan, optimize=optimize)

    if result["status"] == "failed":
        HistoryService.finalize_experiment(exp, status="failed", error=result.get("error"))
        flash(f"Training failed: {result.get('error')}", "error")
        return redirect(url_for("main.builder"))

    # Update experiment
    rec = result.get("recommendation", {})
    HistoryService.update_experiment(
        exp,
        detected_problem=result.get("problem_type"),
        best_model_name=rec.get("model"),
        best_metric_name=rec.get("metric"),
        best_metric_value=rec.get("score"),
        optimized=optimize,
        baseline_performance=rec.get("baseline_score"),
        optimized_performance=rec.get("optimized_score"),
        improvement_pct=rec.get("improvement_pct"),
        overfitting_status=rec.get("overfitting_status"),
        status="completed"
    )
    HistoryService.finalize_experiment(exp, status="completed")

    # Store result in session for display
    flask_session["last_result"] = convert_to_native(result)

    return redirect(url_for("main.results", experiment_id=exp.id))

@main_bp.route("/results/<int:experiment_id>")
@login_required
def results(experiment_id):
    exp = HistoryService.get_experiment(experiment_id, current_user.id)
    if not exp:
        flash("Experiment not found.", "error")
        return redirect(url_for("main.dashboard"))

    result = flask_session.get("last_result")
    if not result or result.get("experiment_id") != exp.public_id if hasattr(exp, "public_id") else True:
        # Fallback: reconstruct minimal result from experiment record
        result = {
            "recommendation": {
                "model": exp.best_model_name,
                "metric": exp.best_metric_name,
                "score": exp.best_metric_value,
            },
            "status": exp.status,
        }

    return render_template("results.html", experiment=exp, result=result)

@main_bp.route("/download/<path:artifact_type>/<path:filename>")
@login_required
def download_artifact(artifact_type, filename):
    usage = UsageService(current_user)
    ok, msg = usage.check_and_increment("download")
    if not ok:
        flash(msg, "error")
        return redirect(url_for("main.dashboard"))

    safe_filename = secure_filename(filename)
    path = os.path.join(current_app.config["ARTIFACT_FOLDER"], safe_filename)
    if not os.path.exists(path):
        flash("File not found.", "error")
        return redirect(url_for("main.dashboard"))

    return send_file(path, as_attachment=True)

@main_bp.route("/history")
@login_required
def history():
    experiments = HistoryService.get_user_experiments(current_user.id, limit=50)
    return render_template("history.html", experiments=experiments)

def read_csv_robust(filepath):
    """Robust CSV reading with multiple encoding fallbacks."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252", "cp850"]
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            df_raw = pd.read_csv(filepath, encoding=enc, dtype=str, keep_default_na=False)
            return df, df_raw, enc
        except Exception:
            continue
    # Final fallback
    df = pd.read_csv(filepath, encoding="utf-8", errors="replace")
    df_raw = pd.read_csv(filepath, encoding="utf-8", errors="replace", dtype=str, keep_default_na=False)
    return df, df_raw, "utf-8-replace"