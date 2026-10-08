"""
Adversarial Validation across PrimeVul
--------------------------------------------------------
"""

import os
import json
import time
import traceback
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE

# =========================================================
# Config -- EDIT THESE PATHS FOR YOUR SETUP
# =========================================================
COMBINED_FILE = "../../../../embedding/primevul/codebert/primevul_embedded.jsonl"
EMB_KEY       = "emb"
RESULTS_PATH  = "results/adversarial_validation_primevul.json"
CACHE_DIR     = "results/oversample_cache"  
SEED          = 42
N_SPLITS      = 5

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

np.random.seed(SEED)

PROJECTS = [
    "linux", "Chrome", "qemu", "gpac",                 # Scenario 1
    "poppler", "radare2_s2", "linux-2.6", "vim", "FFmpeg",   # Scenario 2
    "php-src", "Android", "openssl", "ImageMagick", "tensorflow",  # Scenario 3
    "tcpdump", "radare2_s4", "FreeRDP",                # Scenario 4
]

METHODS = ["No Oversampling", "SMOTE", "ADASYN", "SVMSMOTE"]


# =========================================================
# Result store (load/save with atomic write)
# =========================================================
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_results(results):
    tmp_path = RESULTS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, RESULTS_PATH)  


def result_key(project, method):
    return f"{project}__{method}"


# =========================================================
# Data loading 
# =========================================================
def load_all_embeddings():
    print("Loading combined PrimeVul embeddings (this may take a while)...")
    X, y, projects = [], [], []
    with open(COMBINED_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            X.append(rec[EMB_KEY])
            y.append(rec["target"])
            projects.append(rec["project"])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    projects = np.array(projects, dtype=object)
    print(f"  Loaded {X.shape[0]} total embeddings.")
    return X, y, projects


def build_project_split(X_all, y_all, proj_all, held_out_project):
    """
    Handle the radare2_s2 / radare2_s4 naming: both map back to the
    actual project name 'radare2' in the raw data.
    """
    actual_name = held_out_project.split("_s")[0] if held_out_project.startswith("radare2") else held_out_project

    target_mask = proj_all == actual_name
    source_mask = ~target_mask

    if target_mask.sum() == 0:
        raise ValueError(f"No embeddings found for project '{actual_name}'.")

    source_emb = X_all[source_mask]
    source_lbl = y_all[source_mask]
    target_emb = X_all[target_mask]
    target_lbl = y_all[target_mask]

    return source_emb, source_lbl, target_emb, target_lbl


# =========================================================
# Oversampling with disk caching (per project + method)
# =========================================================
def get_oversampled_source(project, method, source_emb, source_lbl):
    if method == "No Oversampling":
        return source_emb

    cache_path = os.path.join(CACHE_DIR, f"{project}_{method}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    if method == "SMOTE":
        sampler = SMOTE(random_state=SEED)
    elif method == "ADASYN":
        sampler = ADASYN(random_state=SEED)
    elif method == "SVMSMOTE":
        sampler = SVMSMOTE(random_state=SEED)
    else:
        raise ValueError(f"Unknown method: {method}")

    resampled_emb, _ = sampler.fit_resample(source_emb, source_lbl)
    resampled_emb = resampled_emb.astype(np.float32)
    np.save(cache_path, resampled_emb)
    return resampled_emb


# =========================================================
# Adversarial validation core
# =========================================================
def adversarial_validation(source_emb, target_emb, n_splits=N_SPLITS):
    X = np.vstack([source_emb, target_emb])
    y = np.array([0] * len(source_emb) + [1] * len(target_emb))

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    auc_scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, proba))

    return float(np.mean(auc_scores)), float(np.std(auc_scores))


# =========================================================
# Main loop -- project outer, method inner
# =========================================================
def main():
    results = load_results()
    print(f"Loaded {len(results)} existing results from {RESULTS_PATH}\n")

    X_all, y_all, proj_all = load_all_embeddings()

    for project in PROJECTS:
        # Skip entirely if every method for this project is already done
        if all(result_key(project, m) in results for m in METHODS):
            print(f"[{project}] all methods already computed -- skipping project.")
            continue

        print(f"\n=== Project: {project} ===")
        try:
            source_emb, source_lbl, target_emb, target_lbl = build_project_split(
                X_all, y_all, proj_all, project
            )

            scaler = StandardScaler()
            source_emb_scaled = scaler.fit_transform(source_emb).astype(np.float32)
            target_emb_scaled = scaler.transform(target_emb).astype(np.float32)

            print(f"  Source: {source_emb_scaled.shape} | Target: {target_emb_scaled.shape}")

        except Exception as e:
            print(f"  [FATAL for project {project}] Could not build split: {e}")
            traceback.print_exc()
            continue  # move to next project entirely

        for method in METHODS:
            key = result_key(project, method)
            if key in results:
                print(f"  [{method}] already computed -- skipping.")
                continue

            t0 = time.time()
            try:
                print(f"  [{method}] Preparing source embeddings...")
                resampled_source = get_oversampled_source(
                    project, method, source_emb_scaled, source_lbl
                )

                print(f"  [{method}] Running adversarial validation "
                      f"({resampled_source.shape[0]} source vs {target_emb_scaled.shape[0]} target)...")
                mean_auc, std_auc = adversarial_validation(resampled_source, target_emb_scaled)

                results[key] = {
                    "project": project,
                    "method": method,
                    "n_source": int(resampled_source.shape[0]),
                    "n_target": int(target_emb_scaled.shape[0]),
                    "mean_auc": round(mean_auc, 4),
                    "std_auc": round(std_auc, 4),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "status": "ok",
                }
                save_results(results)
                print(f"  [{method}] AUC: {mean_auc:.4f} +/- {std_auc:.4f} "
                      f"({time.time() - t0:.1f}s) -- saved.")

            except Exception as e:
                print(f"  [{method}] FAILED: {e}")
                traceback.print_exc()
                results[key] = {
                    "project": project,
                    "method": method,
                    "status": "failed",
                    "error": str(e),
                    "elapsed_sec": round(time.time() - t0, 1),
                }
                save_results(results)
                continue 

    print("\n=== All projects and methods processed (or already cached). ===")
    print(f"Final results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()