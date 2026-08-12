"""
Cross-Project Vulnerability Detection on PrimeVul
--------------------------------------------------
Domain Adaptation + XGBoost as final classifier
ADASYN Oversampling 
"""

import json
import math
import random
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import ADASYN
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from xgboost import XGBClassifier

# =========================================================
# Config
# =========================================================
SEED            = 42
TEST_PROJS      = ["php-src", "Android", "openssl", "ImageMagick", "tensorflow"]
COMBINED_FILE   = "../../../../embedding/primevul/codebert/primevul_embedded.jsonl"
EMB_KEY         = "emb"
OUTPUT_ROOT     = "results/adasyn"

DANN_EPOCHS          = 50
DANN_BATCH_SIZE      = 64
LEARNING_RATE        = 1e-5
DOMAIN_LOSS_WEIGHT_MAX = 1.0
VALIDATION_SIZE      = 0.2
NUM_RUNS             = 3
THRESHOLD            = 0.5
RESUME               = True
METHOD_VERSION       = "dann_xgboost_adasyn_v1"

ADASYN_N_NEIGHBORS_DEFAULT = 5   

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.manual_seed_all(SEED)
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    torch.mps.manual_seed(SEED)
else:
    DEVICE = "cpu"


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def atomic_json_dump(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
    tmp.replace(path)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    embeddings, labels, projects, sample_ids = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for row_number, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            embeddings.append(record[EMB_KEY])
            labels.append(record["target"])
            projects.append(record["project"])
            sample_ids.append(str(record.get("idx", row_number)))
    return (np.asarray(embeddings, dtype=np.float32),
            np.asarray(labels, dtype=np.int32),
            np.asarray(projects, dtype=object),
            np.asarray(sample_ids, dtype=str))


def apply_adasyn(X_source, y_source, seed):
    classes, counts = np.unique(y_source, return_counts=True)
    minority_count = int(counts.min())

    if minority_count < 2:
        print(f"      [ADASYN] Minority class has only {minority_count} sample(s) -- "
              f"skipping ADASYN, returning original data unchanged.")
        return X_source, y_source

    n_neighbors = min(ADASYN_N_NEIGHBORS_DEFAULT, minority_count - 1)
    if n_neighbors < ADASYN_N_NEIGHBORS_DEFAULT:
        print(f"      [ADASYN] Minority class has only {minority_count} samples -- "
              f"lowering n_neighbors from {ADASYN_N_NEIGHBORS_DEFAULT} to {n_neighbors}.")

    adasyn = ADASYN(random_state=seed, n_neighbors=n_neighbors)
    try:
        X_resampled, y_resampled = adasyn.fit_resample(X_source, y_source)
    except ValueError as exc:
        print(f"      [ADASYN] fit_resample failed ({exc}) -- returning original data unchanged.")
        return X_source, y_source

    X_resampled = X_resampled.astype(np.float32)
    y_resampled = y_resampled.astype(np.int32)

    print(f"      [ADASYN] Source pool: {X_source.shape[0]} -> {X_resampled.shape[0]} samples "
          f"(vulnerable {int((y_source == 1).sum())} -> {int((y_resampled == 1).sum())}, "
          f"benign {int((y_source == 0).sum())} -> {int((y_resampled == 0).sum())})")
    return X_resampled, y_resampled


# =========================================================
# Gradient Reversal Layer
# =========================================================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x, alpha):
        return GradientReversalFunction.apply(x, alpha)


# =========================================================
# Architecture: 64-dim bottleneck FeatureExtractor + linear ClassifierHead
# + DomainDiscriminator
# =========================================================
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.network(x)  # 64-dim domain-invariant features


class ClassifierHead(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.network = nn.Linear(feature_dim, 1)

    def forward(self, features):
        return self.network(features).squeeze(-1)


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


# =========================================================
# DANN training 
# =========================================================
def train_dann(x_source, y_source, x_target, seed):
    set_seed(seed)
    stratify = y_source if len(np.unique(y_source)) > 1 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_source, y_source,
        test_size=VALIDATION_SIZE,
        random_state=seed,
        stratify=stratify,
    )

    feature_extractor = FeatureExtractor(x_source.shape[1]).to(DEVICE)
    classifier         = ClassifierHead(feature_dim=64).to(DEVICE)
    discriminator       = DomainDiscriminator(feature_dim=64).to(DEVICE)
    grl = GradientReversalLayer()

    params = (list(feature_extractor.parameters()) +
              list(classifier.parameters()) +
              list(discriminator.parameters()))
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    classification_loss_fn = nn.BCEWithLogitsLoss()   # no pos_weight -- ADASYN already balances classes (approximately)
    domain_loss_fn = nn.BCEWithLogitsLoss()

    source_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    target_dataset = TensorDataset(torch.tensor(x_target, dtype=torch.float32))
    source_loader = DataLoader(source_dataset, batch_size=DANN_BATCH_SIZE, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=DANN_BATCH_SIZE, shuffle=True)

    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    total_steps = max(1, DANN_EPOCHS * len(source_loader) - 1)
    global_step = 0
    history = []

    for epoch in tqdm(range(DANN_EPOCHS), desc=f"DANN seed={seed}", leave=False):
        feature_extractor.train()
        classifier.train()
        discriminator.train()

        target_iter = iter(target_loader)
        classification_losses, domain_losses = [], []

        for xs_batch, ys_batch in source_loader:
            try:
                (xt_batch,) = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                (xt_batch,) = next(target_iter)

            xs_batch = xs_batch.to(DEVICE)
            ys_batch = ys_batch.to(DEVICE)
            xt_batch = xt_batch.to(DEVICE)

            progress = global_step / total_steps
            alpha = DOMAIN_LOSS_WEIGHT_MAX * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)

            optimizer.zero_grad(set_to_none=True)

            source_features = feature_extractor(xs_batch)
            target_features = feature_extractor(xt_batch)

            vulnerability_logits = classifier(source_features)
            classification_loss = classification_loss_fn(vulnerability_logits, ys_batch)

            source_domain_logits = discriminator(grl(source_features, alpha))
            target_domain_logits = discriminator(grl(target_features, alpha))
            source_domain_labels = torch.zeros(source_domain_logits.shape[0], dtype=torch.float32, device=DEVICE)
            target_domain_labels = torch.ones(target_domain_logits.shape[0], dtype=torch.float32, device=DEVICE)

            source_domain_loss = domain_loss_fn(source_domain_logits, source_domain_labels)
            target_domain_loss = domain_loss_fn(target_domain_logits, target_domain_labels)
            domain_loss = 0.5 * (source_domain_loss + target_domain_loss)

            loss = classification_loss + domain_loss
            loss.backward()
            optimizer.step()

            classification_losses.append(classification_loss.item())
            domain_losses.append(domain_loss.item())
            global_step += 1

        feature_extractor.eval()
        classifier.eval()
        discriminator.eval()
        with torch.no_grad():
            val_logits = classifier(feature_extractor(x_val_t))
            val_loss = classification_loss_fn(val_logits, y_val_t).item()
            val_prob = torch.sigmoid(val_logits)
            val_pred = (val_prob >= THRESHOLD).long().cpu().numpy()
            val_f1 = f1_score(y_val, val_pred, zero_division=0)

        epoch_result = {
            "epoch": epoch + 1,
            "classification_loss": float(np.mean(classification_losses)),
            "domain_loss": float(np.mean(domain_losses)),
            "source_validation_loss": float(val_loss),
            "source_validation_f1": float(val_f1),
            "grl_alpha": float(alpha),
        }
        history.append(epoch_result)
        print(f"      epoch={epoch + 1}/{DANN_EPOCHS} "
              f"cls={epoch_result['classification_loss']:.4f} "
              f"domain={epoch_result['domain_loss']:.4f} "
              f"val={val_loss:.4f} val_f1={val_f1:.4f} alpha={alpha:.4f}")

    return feature_extractor, classifier, discriminator, history


# =========================================================
# XGBoost trained on DANN's extracted (domain-invariant) features.
# =========================================================
def extract_features(feature_extractor, X, batch_size=256):
    feature_extractor.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(DEVICE)
            outputs.append(feature_extractor(x_batch).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, 64), dtype=np.float32)


def classifier_probabilities(feature_extractor, classifier, X, batch_size=256):
    feature_extractor.eval()
    classifier.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(DEVICE)
            logits = classifier(feature_extractor(x_batch))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def train_xgboost(x_train_feat, y_train, sample_weights, seed):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(x_train_feat, y_train, sample_weight=sample_weights)
    return model


# =========================================================
# Evaluation
# =========================================================
def evaluate(y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) > 1 else None
    pr_auc = average_precision_score(y_true, probabilities) if len(np.unique(y_true)) > 1 else None
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    g_mean = math.sqrt(tpr * tnr)
    pf = fp / (fp + tn) if fp + tn else 0.0
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "pr_auc": None if pr_auc is None else float(pr_auc),
        "g_mean": float(g_mean),
        "pf": float(pf),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metrics, predictions


def save_confusion_matrix(metrics, project, path):
    cm_values = metrics["confusion_matrix"]
    cm = np.asarray([[cm_values["tn"], cm_values["fp"]], [cm_values["fn"], cm_values["tp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix ({project}) - DANN + XGBoost + ADASYN")
    plt.colorbar(image, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-Vulnerable", "Vulnerable"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-Vulnerable", "Vulnerable"])
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            color = "black" if cm[i, j] > thresh else "white"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)
    ax.set_ylabel("Actual Label"); ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def cpu_state_dict(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


# =========================================================
# Single run: one DANN training + one XGBoost fit
# =========================================================
def run_single(project, run_number, X_source_adasyn, y_source_adasyn, X_target, y_target,
                target_ids, scaler, output_dir):
    seed = SEED + run_number - 1
    run_dir = Path(output_dir) / f"run_{run_number:02d}"
    result_path = run_dir / "result.json"

    if RESUME and result_path.exists():
        existing = load_json(result_path)
        if existing and existing.get("method_version") == METHOD_VERSION:
            print(f"      Reusing completed run {run_number}: {result_path}")
            return existing

    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = now_utc()

    feature_extractor, classifier, discriminator, history = train_dann(
        X_source_adasyn, y_source_adasyn, X_target, seed
    )

    train_features = extract_features(feature_extractor, X_source_adasyn)
    target_features = extract_features(feature_extractor, X_target)

    train_probs = classifier_probabilities(feature_extractor, classifier, X_source_adasyn)
    sample_weights = np.where(y_source_adasyn == 1, train_probs, 1 - train_probs)

    model_xgb = train_xgboost(train_features, y_source_adasyn, sample_weights, seed)
    probabilities = model_xgb.predict_proba(target_features)[:, 1]

    metrics, predictions = evaluate(y_target, probabilities)
    finished_at = now_utc()

    result = {
        "method_version": METHOD_VERSION,
        "oversampling_method": "ADASYN",
        "project": project,
        "run": run_number,
        "seed": seed,
        "final_classifier": "XGBoost on DANN-extracted features",
        "n_source_samples_adasyn": int(len(y_source_adasyn)),
        "n_target_samples": int(len(y_target)),
        "n_source_vulnerable_adasyn": int(y_source_adasyn.sum()),
        "n_target_vulnerable": int(y_target.sum()),
        "threshold": THRESHOLD,
        "metrics": metrics,
        "training_history": history,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    atomic_json_dump(result, result_path)
    np.savez_compressed(
        run_dir / "predictions.npz",
        idx=target_ids, target=y_target, probability=probabilities, prediction=predictions
    )
    torch.save(
        {
            "method_version": METHOD_VERSION,
            "project": project,
            "run": run_number,
            "seed": seed,
            "feature_extractor": cpu_state_dict(feature_extractor),
            "classifier": cpu_state_dict(classifier),
            "domain_discriminator": cpu_state_dict(discriminator),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "threshold": THRESHOLD,
        },
        run_dir / "model.pt",
    )
    save_confusion_matrix(metrics, project, run_dir / "confusion_matrix.png")
    print(f"      Saved run {run_number} to {run_dir}")

    del feature_extractor, classifier, discriminator, model_xgb
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()

    return result


# =========================================================
# Ensemble aggregation
# =========================================================
def aggregate_ensemble(project, run_results, output_dir, y_target):
    prob_arrays = []
    for r in run_results:
        npz_path = Path(output_dir) / f"run_{r['run']:02d}" / "predictions.npz"
        data = np.load(npz_path)
        prob_arrays.append(data["probability"])

    ensemble_probability = np.mean(np.stack(prob_arrays, axis=0), axis=0)
    ensemble_metrics, ensemble_predictions = evaluate(y_target, ensemble_probability)

    per_run_metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "g_mean", "pf"]
    per_run_summary = {}
    for metric in per_run_metric_names:
        values = [r["metrics"][metric] for r in run_results if r["metrics"][metric] is not None]
        per_run_summary[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None),
        }

    return {
        "method_version": METHOD_VERSION,
        "method": "DANN + XGBoost + ADASYN",
        "project": project,
        "train": f"all_except_{project}",
        "num_completed_runs": len(run_results),
        "ensemble_metrics": ensemble_metrics,
        "per_run_metric_summary": per_run_summary,
        "runs": run_results,
        "updated_at": now_utc(),
    }


# =========================================================
# Per-project orchestration
# =========================================================
def run_project(project, X_all, y_all, projects_all, ids_all,
                 master_results, master_path, state, state_path):
    project_key = safe_name(project)
    output_dir = Path(OUTPUT_ROOT) / project_key
    output_dir.mkdir(parents=True, exist_ok=True)

    source_mask = projects_all != project
    target_mask = projects_all == project
    if not target_mask.any():
        state["skipped"][project] = {"reason": "No target samples found", "time": now_utc()}
        atomic_json_dump(state, state_path)
        print(f"[SKIPPED] {project}: no samples found")
        return None

    X_source = X_all[source_mask].astype(np.float32)
    y_source = y_all[source_mask].astype(np.int32)
    X_target = X_all[target_mask].astype(np.float32)
    y_target = y_all[target_mask].astype(np.int32)
    target_ids = ids_all[target_mask]

    if len(np.unique(y_source)) < 2:
        raise ValueError(f"Source pool for {project} does not contain both classes.")

    scaler = StandardScaler()
    X_source = scaler.fit_transform(X_source).astype(np.float32)
    X_target = scaler.transform(X_target).astype(np.float32)

    source_projects = len(set(projects_all[source_mask].tolist()))
    print("\n" + "=" * 88)
    print(f"Target project: {project}")
    print(f"Source projects: {source_projects}")
    print(f"Source (pre-ADASYN): {X_source.shape} vulnerable={int(y_source.sum())} benign={int((y_source == 0).sum())}")
    print(f"Target: {X_target.shape} vulnerable={int(y_target.sum())} benign={int((y_target == 0).sum())}")
    print("=" * 88)

    X_source_adasyn, y_source_adasyn = apply_adasyn(X_source, y_source, seed=SEED)

    run_results = []
    for run_number in range(1, NUM_RUNS + 1):
        print(f"\n      Run {run_number}/{NUM_RUNS}")
        try:
            result = run_single(
                project, run_number, X_source_adasyn, y_source_adasyn, X_target, y_target,
                target_ids, scaler, output_dir
            )
            run_results.append(result)
        except Exception as exc:
            print(f"      [RUN FAILED] {project} run {run_number}: {exc}")
            state["in_progress"].setdefault(project, {})["last_run_error"] = {
                "run": run_number, "error": str(exc), "traceback": traceback.format_exc(), "time": now_utc()
            }
            atomic_json_dump(state, state_path)
            continue

        if run_results:
            ensemble_summary = aggregate_ensemble(project, run_results, output_dir, y_target)
            atomic_json_dump(ensemble_summary, output_dir / "project_summary.json")
            master_results[project] = ensemble_summary
            master_payload = {
                "method_version": METHOD_VERSION,
                "method": "DANN + XGBoost + ADASYN",
                "updated_at": now_utc(),
                "projects": master_results,
            }
            atomic_json_dump(master_payload, master_path)

        state["in_progress"][project] = {
            "completed_runs": len(run_results), "total_runs": NUM_RUNS, "time": now_utc()
        }
        atomic_json_dump(state, state_path)

    if not run_results:
        raise RuntimeError(f"All {NUM_RUNS} runs failed for project {project}.")

    state["in_progress"].pop(project, None)
    state["completed"][project] = {"runs": len(run_results), "time": now_utc()}
    state["failed"].pop(project, None)
    atomic_json_dump(state, state_path)
    return master_results[project]


# =========================================================
# Main: multi-project loop
# =========================================================
def main():
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    master_path = output_root / "dann_xgboost_adasyn_all_projects.json"
    state_path = output_root / "run_state.json"

    existing_master = load_json(master_path, {}) if RESUME else {}
    master_results = existing_master.get("projects", {}) if existing_master.get("method_version") == METHOD_VERSION else {}

    existing_state = load_json(state_path, {}) if RESUME else {}
    if existing_state.get("method_version") == METHOD_VERSION:
        state = existing_state
        state.setdefault("completed", {})
        state.setdefault("failed", {})
        state.setdefault("skipped", {})
        state.setdefault("in_progress", {})
    else:
        state = {
            "method_version": METHOD_VERSION,
            "requested_projects": TEST_PROJS,
            "completed": {}, "failed": {}, "skipped": {}, "in_progress": {},
            "started_at": now_utc(),
        }
    atomic_json_dump(state, state_path)

    print(f"Device: {DEVICE}")
    print(f"Method: {METHOD_VERSION}")
    print(f"Projects: {TEST_PROJS}")
    print(f"Dataset: {COMBINED_FILE}")

    X_all, y_all, projects_all, ids_all = load_jsonl(COMBINED_FILE)
    print(f"Loaded {len(y_all)} samples from {len(set(projects_all.tolist()))} projects")

    for project_index, project in enumerate(TEST_PROJS, start=1):
        if RESUME and project in state["completed"]:
            print(f"\nProject {project_index}/{len(TEST_PROJS)}: {project} -- already completed, skipping")
            continue

        print(f"\nProject {project_index}/{len(TEST_PROJS)}: {project}")
        try:
            run_project(project, X_all, y_all, projects_all, ids_all,
                        master_results, master_path, state, state_path)
        except KeyboardInterrupt:
            state["in_progress"][project] = {"interrupted": True, "time": now_utc()}
            atomic_json_dump(state, state_path)
            raise
        except Exception as exc:
            state["in_progress"].pop(project, None)
            state["failed"][project] = {
                "error": str(exc), "traceback": traceback.format_exc(), "time": now_utc()
            }
            atomic_json_dump(state, state_path)
            print(f"[FAILED] {project}: {exc}")
            continue

    final_payload = {
        "method_version": METHOD_VERSION,
        "method": "DANN + XGBoost + ADASYN",
        "updated_at": now_utc(),
        "projects": master_results,
    }
    atomic_json_dump(final_payload, master_path)
    state["finished_at"] = now_utc()
    atomic_json_dump(state, state_path)

    print(f"\nResults: {master_path}")
    print(f"State: {state_path}")
    print(f"Completed: {list(state['completed'].keys())}")
    print(f"Failed: {list(state['failed'].keys())}")
    print(f"Skipped: {list(state['skipped'].keys())}")


if __name__ == "__main__":
    main()