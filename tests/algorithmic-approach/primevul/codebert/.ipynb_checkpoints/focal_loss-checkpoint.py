"""
Cross-Project Vulnerability Detection
--------------------------------------
PrimeVul Scenarios 1-4 (Leave-One-Project-Out), all held-out projects
Train on ALL projects EXCEPT the test project -> Test on that project
Focal Loss (same configuration as the ReVeal / PrimeVul-linux scripts)
"""

import os
import gc
import json
import time
import random
import warnings
import traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    roc_auc_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =========================================================
# Config
# =========================================================
SEED = 42

COMBINED_FILE = "../../../../embedding/primevul/codebert/primevul_embedded.jsonl"
EMB_KEY       = "emb"
RESULTS_PATH  = "results/focal_loss/primevul_all_projects.json"

FOCAL_ALPHA = 0.85
FOCAL_GAMMA = 2.0

NUM_ITERATIONS = 3
LOG_EVERY      = 10    

PROJECTS = [
    (1, "linux"), (1, "Chrome"), (1, "qemu"), (1, "gpac"),
    (2, "poppler"), (2, "radare2"), (2, "linux-2.6"), (2, "vim"), (2, "FFmpeg"),
    (3, "php-src"), (3, "Android"), (3, "openssl"), (3, "ImageMagick"), (3, "tensorflow"),
    (4, "tcpdump"), (4, "FreeRDP"),
]
DUPLICATE_PROJECT = "radare2"  

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc", "g_mean", "pf"]

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "mps":
        torch.mps.manual_seed(seed)
    elif DEVICE == "cuda":
        torch.cuda.manual_seed(seed)


def free_memory():
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


# =========================================================
# Results store
# =========================================================
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    tmp_path = RESULTS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, RESULTS_PATH)


def print_progress(results):
    done   = sum(1 for _, p in PROJECTS if results.get(p, {}).get("status") == "ok")
    failed = sum(1 for _, p in PROJECTS if results.get(p, {}).get("status") == "failed")
    total  = len(PROJECTS)
    print(f"----- PROGRESS: {done} ok | {failed} failed | "
          f"{total - done - failed} remaining | {100 * (done + failed) / total:.1f}% "
          f"of {total} projects -----")


# =========================================================
# Data
# =========================================================
def load_jsonl(path):
    X, y, projects = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            X.append(r[EMB_KEY])
            y.append(r["target"])
            projects.append(r["project"])
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(projects, dtype=object))


# =========================================================
# Model
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        bce     = nn.functional.binary_cross_entropy(
                      predictions, targets, reduction='none'
                  )
        p_t     = predictions * targets + (1 - predictions) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss    = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class VulnerabilityClassifier(nn.Module):
    def __init__(self, input_dim):
        super(VulnerabilityClassifier, self).__init__()
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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_neural_network(model, dataloader, x_val, y_val,
                         optimizer, criterion, epochs, desc):
    for epoch in tqdm(range(epochs), desc=desc, leave=False):
        model.train()
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(x_batch).squeeze(-1)
            loss        = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % LOG_EVERY == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_preds = model(x_val).squeeze(-1)
                val_loss  = criterion(val_preds, y_val).item()
            print(f"        {desc} | Epoch {epoch+1}/{epochs} "
                  f"- val_loss: {val_loss:.4f}")

    return model


def semi_supervised_transfer_learning(x_train, y_train, x_test, iteration):
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train,
        test_size=0.2,
        random_state=SEED
    )

    x_tr_t   = torch.tensor(x_tr,   dtype=torch.float32)
    y_tr_t   = torch.tensor(y_tr,   dtype=torch.float32)
    x_val_t  = torch.tensor(x_val,  dtype=torch.float32).to(DEVICE)
    y_val_t  = torch.tensor(y_val,  dtype=torch.float32).to(DEVICE)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)

    dataset    = TensorDataset(x_tr_t, y_tr_t)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model     = VulnerabilityClassifier(x_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    model = train_neural_network(
        model, dataloader, x_val_t, y_val_t,
        optimizer, criterion,
        epochs=50,
        desc=f"Iter {iteration} - Phase 1"
    )

    model.eval()
    with torch.no_grad():
        y_pred        = model(x_test_t.to(DEVICE)).squeeze(-1).cpu().numpy()
        y_pred_binary = (y_pred > 0.5).astype(int)

    x_train_aug    = np.concatenate((x_train, x_test))
    y_train_aug    = np.concatenate((y_train, y_pred_binary))
    x_aug_t        = torch.tensor(x_train_aug, dtype=torch.float32)
    y_aug_t        = torch.tensor(y_train_aug, dtype=torch.float32)
    x_aug_val_t    = torch.tensor(x_val,  dtype=torch.float32).to(DEVICE)
    y_aug_val_t    = torch.tensor(y_val,  dtype=torch.float32).to(DEVICE)
    dataset_aug    = TensorDataset(x_aug_t, y_aug_t)
    dataloader_aug = DataLoader(dataset_aug, batch_size=32, shuffle=True)

    model = train_neural_network(
        model, dataloader_aug, x_aug_val_t, y_aug_val_t,
        optimizer, criterion,
        epochs=30,
        desc=f"Iter {iteration} - Phase 2"
    )

    model.eval()
    with torch.no_grad():
        y_pred_final = model(x_test_t.to(DEVICE)).squeeze(-1).cpu().numpy()

    return model, y_pred_final


def train_base_model_xgb(x_train, y_train, n_neg, n_pos, sample_weights=None):
    scale_pos_weight = n_neg / n_pos
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    model.fit(x_train, y_train, sample_weight=sample_weights)
    return model


# =========================================================
# One held-out project
# =========================================================
def run_project(scenario, project, X_all, y_all, proj_all):
    train_mask = proj_all != project
    test_mask  = proj_all == project

    if test_mask.sum() == 0:
        raise ValueError(
            f"No samples found for project '{project}'. "
            f"Available names (first 30): {sorted(set(proj_all))[:30]}"
        )

    train_emb, train_lbl = X_all[train_mask], y_all[train_mask]
    test_emb,  test_lbl  = X_all[test_mask],  y_all[test_mask]

    n_neg = int(np.sum(train_lbl == 0))
    n_pos = int(np.sum(train_lbl == 1))
    n_train_projects = len(set(proj_all[train_mask]))
    print(f"    Train: {train_emb.shape} (0: {n_neg} | 1: {n_pos}, "
          f"{n_train_projects} projects) | Test: {test_emb.shape} "
          f"(vuln: {int(test_lbl.sum())})")

    scaler    = StandardScaler()
    train_emb = scaler.fit_transform(train_emb)
    test_emb  = scaler.transform(test_emb)

    ensemble_predictions = np.zeros(len(test_lbl))

    for i in range(NUM_ITERATIONS):
        print(f"\n      [Iteration {i+1}/{NUM_ITERATIONS}] Training neural network...")
        model_nn, _ = semi_supervised_transfer_learning(
            train_emb, train_lbl, test_emb, iteration=i+1
        )

        print(f"      [Iteration {i+1}/{NUM_ITERATIONS}] Training XGBoost...")
        model_nn.eval()
        with torch.no_grad():
            x_tr_t       = torch.tensor(train_emb, dtype=torch.float32).to(DEVICE)
            y_train_pred = model_nn(x_tr_t).squeeze(-1).cpu().numpy()

        sample_weights = np.where(train_lbl == 1, y_train_pred, 1 - y_train_pred)
        model_xgb      = train_base_model_xgb(
                             train_emb, train_lbl, n_neg, n_pos, sample_weights
                         )
        ensemble_predictions += model_xgb.predict_proba(test_emb)[:, 1]
        print(f"      [Iteration {i+1}/{NUM_ITERATIONS}] Done")

        del model_nn, model_xgb, x_tr_t
        free_memory()

    ensemble_avg = ensemble_predictions / NUM_ITERATIONS
    y_pred_final = (ensemble_avg > 0.5).astype(int)

    accuracy  = accuracy_score(test_lbl, y_pred_final)
    recall    = recall_score(test_lbl, y_pred_final, zero_division=0)
    precision = precision_score(test_lbl, y_pred_final, zero_division=0)
    auc       = roc_auc_score(test_lbl, ensemble_avg)
    f1        = f1_score(test_lbl, y_pred_final, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(test_lbl, y_pred_final, labels=[0, 1]).ravel()
    g_mean = np.sqrt((tp / (tp + fn + 1e-9)) * (tn / (tn + fp + 1e-9)))
    pf     = fp / (fp + tn + 1e-9)

    return {
        "scenario"         : scenario,
        "project"          : project,
        "focal_alpha"      : FOCAL_ALPHA,
        "focal_gamma"      : FOCAL_GAMMA,
        "scale_pos_weight" : round(n_neg / n_pos, 3),
        "n_train_projects" : int(n_train_projects),
        "n_train_samples"  : int(len(train_lbl)),
        "n_test_samples"   : int(len(test_lbl)),
        "n_vuln_train"     : int(n_pos),
        "n_vuln_test"      : int(test_lbl.sum()),
        "accuracy"         : round(float(accuracy),  4),
        "precision"        : round(float(precision), 4),
        "recall"           : round(float(recall),    4),
        "f1"               : round(float(f1),        4),
        "auc"              : round(float(auc),       4),
        "g_mean"           : round(float(g_mean),    4),
        "pf"               : round(float(pf),        4),
        "confusion_matrix" : {"tn": int(tn), "fp": int(fp),
                              "fn": int(fn), "tp": int(tp)},
    }


# =========================================================
# Final summary
# =========================================================
def print_summary(results):
    rows = [results[p] for _, p in PROJECTS
            if results.get(p, {}).get("status") == "ok"]

    width = 118
    print("\n" + "=" * width)
    print(f"FINAL RESULTS - PrimeVul Scenarios 1-4 | Focal Loss "
          f"(alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
    print("=" * width)
    print(f"{'Scn':<4}{'Project':<13}{'Test':>8}{'Vuln':>7}{'Acc':>8}{'Prec':>8}"
          f"{'Recall':>8}{'F1':>8}{'AUC':>8}{'G-mean':>8}{'PF':>8}"
          f"{'TN':>8}{'FP':>7}{'FN':>6}{'TP':>6}")
    print("-" * width)
    for r in rows:
        cm = r["confusion_matrix"]
        print(f"{r['scenario']:<4}{r['project']:<13}{r['n_test_samples']:>8}"
              f"{r['n_vuln_test']:>7}{r['accuracy']:>8.3f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}{r['auc']:>8.3f}"
              f"{r['g_mean']:>8.3f}{r['pf']:>8.3f}"
              f"{cm['tn']:>8}{cm['fp']:>7}{cm['fn']:>6}{cm['tp']:>6}")
    print("-" * width)

    if rows:
        mean_unique = {k: np.mean([r[k] for r in rows]) for k in METRIC_KEYS}
        line = "  ".join(f"{k}={mean_unique[k]:.3f}" for k in METRIC_KEYS)
        print(f"Mean over {len(rows)} unique projects: {line}")

        dup = results.get(DUPLICATE_PROJECT, {})
        if dup.get("status") == "ok":
            n_entries = len(rows) + 1
            mean_dup = {k: (sum(r[k] for r in rows) + dup[k]) / n_entries
                        for k in METRIC_KEYS}
            line_dup = "  ".join(f"{k}={mean_dup[k]:.3f}" for k in METRIC_KEYS)
            print(f"Mean over {n_entries} entries ({DUPLICATE_PROJECT} in S2 and S4, "
                  f"as in RQ1): {line_dup}")

    failed  = [p for _, p in PROJECTS if results.get(p, {}).get("status") == "failed"]
    missing = [p for _, p in PROJECTS if p not in results]
    if failed:
        print(f"\nFAILED projects (re-run the script to retry): {failed}")
    if missing:
        print(f"NOT YET RUN: {missing}")
    print("=" * width)
    print(f"Results file: {RESULTS_PATH}")


# =========================================================
# Main
# =========================================================
def main():
    print("\n=== PrimeVul S1-4 | Leave-One-Project-Out | Focal Loss ===")
    print(f"    Device: {DEVICE}")
    print(f"    Focal Loss - alpha: {FOCAL_ALPHA} | gamma: {FOCAL_GAMMA}")

    results = load_results()
    print(f"    Loaded {len(results)} existing results from {RESULTS_PATH}")
    print_progress(results)

    pending = [(s, p) for s, p in PROJECTS
               if results.get(p, {}).get("status") != "ok"]
    if not pending:
        print("\nAll projects already computed.")
        print_summary(results)
        return

    print("\n[1/2] Loading embeddings...")
    X_all, y_all, proj_all = load_jsonl(COMBINED_FILE)
    print(f"      Loaded {X_all.shape[0]} samples, "
          f"{len(set(proj_all))} projects in total")

    print("\n[2/2] Running held-out projects...")
    for idx, (scenario, project) in enumerate(PROJECTS, start=1):
        if results.get(project, {}).get("status") == "ok":
            r = results[project]
            print(f"\n[{idx}/{len(PROJECTS)}] S{scenario} {project}: already computed "
                  f"(F1={r['f1']:.3f}) - skipping.")
            continue

        print(f"\n[{idx}/{len(PROJECTS)}] S{scenario} - held-out project: {project}")
        t0 = time.time()
        set_seed(SEED)

        try:
            metrics = run_project(scenario, project, X_all, y_all, proj_all)
            metrics["elapsed_sec"] = round(time.time() - t0, 1)
            metrics["status"] = "ok"
            results[project] = metrics
            save_results(results)
            print(f"\n    {project}: Acc={metrics['accuracy']:.3f} "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                  f"F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f} "
                  f"G={metrics['g_mean']:.3f} PF={metrics['pf']:.3f} "
                  f"({metrics['elapsed_sec']}s) - saved.")
        except Exception as e:
            print(f"\n    FAILED on {project}: {e}")
            traceback.print_exc()
            results[project] = {
                "scenario": scenario, "project": project,
                "status": "failed", "error": str(e),
                "elapsed_sec": round(time.time() - t0, 1),
            }
            save_results(results)
        finally:
            free_memory()

        print_progress(results)

    print_summary(results)


if __name__ == "__main__":
    main()