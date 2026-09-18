"""Main application routes: dashboard, upload, AutoML builder, results."""
import os
import uuid
import json
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, send_file, session as flask_session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from models.user import db
from services.usage_service import UsageService
from services.history_service import HistoryService
from services.subscription_service import SubscriptionService
from ml.pipeline import MLPilotPipeline
from ml.detection import analyze_dataset, suggest_target_columns, detect_problem_type, recommend_algorithms
from ml.models.classification import CLASSIFICATION_MODELS
from ml.models.regression import REGRESSION_MODELS

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

        try:
            df, df_raw, encoding = read_csv_robust(filepath)
        except Exception as e:
            flash(f"Could not read file: {str(e)}", "error")
            return redirect(url_for("main.builder"))

        cfg = current_user.get_plan_config(current_app.config)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > cfg.get("max_dataset_size_mb", 10):
            flash(f"Dataset too large ({size_mb:.1f} MB). Plan limit: {cfg['max_dataset_size_mb']} MB.", "error")
            return redirect(url_for("main.builder"))

        analysis = analyze_dataset(df)
        suggested_targets = suggest_target_columns(df, analysis)

        # Build rich column metadata for manual mode feature table
        column_types = {}
        column_uniques = {}
        column_samples = {}
        for col in df.columns:
            col_info = analysis["columns"].get(col, {})
            column_types[col] = col_info.get("type", "other")
            column_uniques[col] = col_info.get("unique", 0)
            sample_vals = df[col].dropna().astype(str).head(3).tolist()
            column_samples[col] = ", ".join(sample_vals) if sample_vals else "—"

        # Only the filename needs to survive the redirect — the cookie is 4KB.
        # analysis/column_types/column_uniques/column_samples are passed straight
        # into the template below and re-derived in train(), so keeping them in
        # the session only inflates the cookie until Flask silently drops it.
        flask_session["current_dataset"] = unique_name

        # Available models for manual picker
        classification_models = list(CLASSIFICATION_MODELS.keys())
        regression_models = list(REGRESSION_MODELS.keys())

        return render_template("builder.html",
                               dataset_name=file.filename,
                               dataset_path=unique_name,
                               analysis=analysis,
                               suggested_targets=suggested_targets,
                               columns=df.columns.tolist(),
                               encoding=encoding,
                               column_types=column_types,
                               column_uniques=column_uniques,
                               column_samples=column_samples,
                               classification_models=classification_models,
                               regression_models=regression_models)

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
    selected_features = request.form.getlist("features") if run_mode == "manual" else None
    selected_models = request.form.getlist("models") if run_mode == "manual" else None

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

    # Manual mode: filter features
    if run_mode == "manual" and selected_features:
        keep_cols = selected_features[:]
        if target_col and target_col in df.columns and target_col not in keep_cols:
            keep_cols.append(target_col)
        df = df[keep_cols]

    # Create experiment record
    exp = HistoryService.create_experiment(
        current_user.id,
        name=f"Experiment {dataset_name[:20]}",
        dataset_name=dataset_name
    )

    # Run pipeline with bulletproof error handling
    try:
        pipeline = MLPilotPipeline(artifact_folder=current_app.config.get("ARTIFACT_FOLDER", "artifacts"))
        result = pipeline.run(df, target_col=target_col, user_plan=current_user.plan, optimize=optimize)
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        current_app.logger.error(f"Pipeline error: {str(e)}\n{error_detail}")
        HistoryService.finalize_experiment(exp, status="failed", error=f"{str(e)} — {error_detail[:500]}")
        flash(f"Training failed: {str(e)}. This dataset may have incompatible features. Try a different dataset or check data quality.", "error")
        return redirect(url_for("main.builder"))

    if result["status"] == "failed":
        HistoryService.finalize_experiment(exp, status="failed", error=result.get("error"))
        flash(f"Training failed: {result.get('error')}", "error")
        return redirect(url_for("main.builder"))

    # Manual mode: filter results to selected models + ensemble
    if run_mode == "manual" and selected_models:
        filtered_results = []
        filtered_model_paths = {}
        for r in result.get("results", []):
            base_name = r["model"].replace(" (Optimized)", "")
            if base_name in selected_models or r["model"] in selected_models or "Ensemble" in r["model"]:
                filtered_results.append(r)
                if r["model"] in result.get("model_paths", {}):
                    filtered_model_paths[r["model"]] = result["model_paths"][r["model"]]
        # Always keep ensemble if it exists and >=2 selected
        for r in result.get("results", []):
            if "Ensemble" in r["model"] and r not in filtered_results:
                filtered_results.append(r)
                if r["model"] in result.get("model_paths", {}):
                    filtered_model_paths[r["model"]] = result["model_paths"][r["model"]]
        result["results"] = filtered_results
        result["model_paths"] = {**filtered_model_paths, **{k: v for k, v in result.get("model_paths", {}).items() if "Ensemble" in k}}
        # Re-rank
        problem_type = result.get("problem_type", "classification")
        if problem_type == "classification":
            result["results"].sort(key=lambda x: x.get("primary_score", 0), reverse=True)
        else:
            result["results"].sort(key=lambda x: x.get("primary_score", float("inf")))
        # Update recommendation
        if result["results"]:
            best = result["results"][0]
            result["recommendation"] = {
                "model": best["model"],
                "problem_type": problem_type,
                "metric": "Accuracy" if problem_type == "classification" else "RMSE",
                "score": best.get("primary_score"),
                "baseline_score": best.get("baseline_score"),
                "optimized_score": best.get("optimized_score"),
                "improvement_pct": best.get("improvement_pct"),
                "overfitting_status": best.get("diagnostics", {}).get("status"),
                "preprocessing_applied": result.get("preprocessing", {}).get("cleaning_log", []),
                "smote_applied": result.get("smote_applied", False),
                "pca_applied": result.get("preprocessing", {}).get("pca_applied", False),
            }

    # Create production bundle ZIP
    bundle_path = None
    try:
        bundle_path = pipeline.artifact_manager.package_all(
            pipeline.experiment_id,
            result.get("model_paths", {}),
            result.get("preproc_paths", {}),
            result.get("metadata", {})
        )
    except Exception as e:
        current_app.logger.warning(f"Bundle packaging failed: {e}")

    # Update experiment with full result persistence
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
        status="completed",
        run_mode=run_mode,
        target_column=target_col,
        dataset_shape=result.get("metadata", {}).get("dataset_shape"),
    )

    # Persist full result JSON for reconstruction
    result_native = convert_to_native(result)
    if bundle_path:
        result_native["bundle_path"] = bundle_path
    HistoryService.set_result_json(exp, json.dumps(result_native))
    HistoryService.set_preprocessing_log(exp, json.dumps(result.get("preprocessing", {}).get("cleaning_log", [])))
    HistoryService.set_models_tested(exp, json.dumps(result.get("results", [])))
    HistoryService.set_artifact_paths(exp, json.dumps({
        "model_paths": result.get("model_paths", {}),
        "preproc_paths": result.get("preproc_paths", {}),
        "bundle_path": bundle_path,
    }))
    HistoryService.set_features_used(exp, json.dumps(result.get("preprocessing", {}).get("feature_names", [])))
    HistoryService.finalize_experiment(exp, status="completed")

    # Result is persisted in exp.result_json; the session cookie is 4KB and
    # cannot hold a full result payload without being silently dropped.
    flask_session["last_experiment_id"] = exp.id
    flask_session.pop("last_result", None)

    return redirect(url_for("main.results", experiment_id=exp.id))


@main_bp.route("/results/<int:experiment_id>")
@login_required
def results(experiment_id):
    exp = HistoryService.get_experiment(experiment_id, current_user.id)
    if not exp:
        flash("Experiment not found.", "error")
        return redirect(url_for("main.dashboard"))

    result = reconstruct_result_from_experiment(exp)

    artifact_folder = current_app.config.get("ARTIFACT_FOLDER", "artifacts")
    return render_template("results.html", experiment=exp, result=result, artifact_folder=artifact_folder)


def reconstruct_result_from_experiment(exp):
    """Reconstruct full result dict from experiment DB record."""
    result = {
        "status": exp.status,
        "experiment_id": getattr(exp, "experiment_id", None),
        "problem_type": exp.detected_problem,
        "smote_applied": False,
        "preprocessing": {},
        "results": [],
        "recommendation": {
            "model": exp.best_model_name,
            "metric": exp.best_metric_name,
            "score": exp.best_metric_value,
            "baseline_score": exp.baseline_performance,
            "optimized_score": exp.optimized_performance,
            "improvement_pct": exp.improvement_pct,
            "overfitting_status": exp.overfitting_status,
        },
        "model_paths": {},
        "preproc_paths": {},
        "metadata": {
            "dataset_shape": getattr(exp, "dataset_shape", None),
            "target_column": getattr(exp, "target_column", None),
        },
    }

    # Reconstruct from JSON fields
    if exp.result_json:
        try:
            stored = json.loads(exp.result_json)
            result.update(stored)
        except Exception:
            pass

    if exp.models_tested:
        try:
            result["results"] = json.loads(exp.models_tested)
        except Exception:
            pass

    if exp.artifact_paths:
        try:
            paths = json.loads(exp.artifact_paths)
            result["model_paths"] = paths.get("model_paths", {})
            result["preproc_paths"] = paths.get("preproc_paths", {})
            result["bundle_path"] = paths.get("bundle_path")
        except Exception:
            pass

    if exp.preprocessing_log:
        try:
            logs = json.loads(exp.preprocessing_log)
            result["preprocessing"]["cleaning_log"] = logs
            result["smote_applied"] = any("SMOTE" in str(log) for log in logs)
        except Exception:
            pass

    if exp.features_used:
        try:
            result["preprocessing"]["feature_names"] = json.loads(exp.features_used)
        except Exception:
            pass

    return result


@main_bp.route("/download/<path:artifact_type>/<path:filename>")
@login_required
def download_artifact(artifact_type, filename):
    usage = UsageService(current_user)
    ok, msg = usage.check_and_increment("download")
    if not ok:
        flash(msg, "error")
        return redirect(url_for("main.dashboard"))

    # The filename from template may be a relative path like "exp_id/model.pkl"
    # or an absolute path. We need to resolve it correctly.
    artifact_folder = current_app.config.get("ARTIFACT_FOLDER", "artifacts")

    # Try multiple path resolution strategies
    possible_paths = []

    # Strategy 1: filename is already relative to project root (contains artifact_folder)
    if filename.startswith(artifact_folder):
        possible_paths.append(filename)

    # Strategy 2: join artifact_folder with filename
    possible_paths.append(os.path.join(artifact_folder, filename))

    # Strategy 3: filename might have backslashes on Windows
    possible_paths.append(os.path.join(artifact_folder, filename.replace("/", os.sep)))

    # Strategy 4: try with secure_filename
    safe = secure_filename(filename)
    possible_paths.append(os.path.join(artifact_folder, safe))

    # Strategy 5: the path might already be absolute
    if os.path.isabs(filename):
        possible_paths.append(filename)

    # Find the first existing path
    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break

    if not path:
        current_app.logger.error(f"Artifact not found. Tried: {possible_paths}")
        flash(f"File not found: {filename}. The artifact may not have been generated for this experiment.", "error")
        return redirect(url_for("main.dashboard"))

    return send_file(path, as_attachment=True)


@main_bp.route("/download/bundle/<int:experiment_id>")
@login_required
def download_bundle(experiment_id):
    """Download the unified production package for an experiment."""
    usage = UsageService(current_user)
    ok, msg = usage.check_and_increment("download")
    if not ok:
        flash(msg, "error")
        return redirect(url_for("main.results", experiment_id=experiment_id))

    exp = HistoryService.get_experiment(experiment_id, current_user.id)
    if not exp:
        flash("Experiment not found.", "error")
        return redirect(url_for("main.dashboard"))

    # Try to get bundle path from artifact_paths
    bundle_path = None
    if exp.artifact_paths:
        try:
            paths = json.loads(exp.artifact_paths)
            bundle_path = paths.get("bundle_path")
        except Exception:
            pass

    if not bundle_path or not os.path.exists(bundle_path):
        flash("Production bundle not found for this experiment.", "error")
        return redirect(url_for("main.results", experiment_id=experiment_id))

    return send_file(bundle_path, as_attachment=True, download_name=f"MLPilot_Production_Package_{experiment_id}.zip")


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
    df = pd.read_csv(filepath, encoding="utf-8", errors="replace")
    df_raw = pd.read_csv(filepath, encoding="utf-8", errors="replace", dtype=str, keep_default_na=False)
    return df, df_raw, "utf-8-replace"