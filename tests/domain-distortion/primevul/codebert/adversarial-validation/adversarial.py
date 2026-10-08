"""
Adversarial Validation across PrimeVul
--------------------------------------------------------
"""

import os
import gc
import json
import time
import traceback
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE


COMBINED_FILE = "../../../../../embedding/primevul/codebert/primevul_embedded.jsonl"
EMB_KEY       = "emb"
RESULTS_PATH  = "results/adversarial_validation_primevul.json"
CACHE_DIR     = "results/oversample_cache"
SEED          = 42
N_SPLITS      = 5

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

np.random.seed(SEED)

PROJECTS = [
    "linux", "Chrome", "qemu", "gpac",
    "poppler", "radare2_s2", "linux-2.6", "vim", "FFmpeg",
    "php-src", "Android", "openssl", "ImageMagick", "tensorflow",
    "tcpdump", "radare2_s4", "FreeRDP",
]

METHODS = ["No Oversampling", "SMOTE", "ADASYN", "SVMSMOTE"]

TOTAL_RUNS = len(PROJECTS) * len(METHODS)


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


def print_progress_summary(results):
    done = sum(1 for v in results.values() if v.get("status") == "ok")
    failed = sum(1 for v in results.values() if v.get("status") == "failed")
    remaining = TOTAL_RUNS - done - failed
    pct = 100 * (done + failed) / TOTAL_RUNS
    print(f"----- PROGRESS: {done} ok | {failed} failed | {remaining} remaining "
          f"| {pct:.1f}% of {TOTAL_RUNS} total runs -----")


def load_all_embeddings():
    t0 = time.time()
    X, y, projects = [], [], []
    with open(COMBINED_FILE, "r") as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Parsing JSONL", unit="rec"):
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
    print(f"Loaded {X.shape[0]} embeddings in {time.time() - t0:.1f}s "
          f"({X.nbytes / 1e9:.2f} GB)")
    return X, y, projects


def build_project_split(X_all, y_all, proj_all, held_out_project):
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


def adversarial_validation(source_emb, target_emb, label="", n_splits=N_SPLITS):
    X = np.vstack([source_emb, target_emb])
    y = np.array([0] * len(source_emb) + [1] * len(target_emb))

    n_rows = X.shape[0]
    print(f"    Fitting on {n_rows} rows (HistGradientBoosting, fast on large n)...")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    auc_scores = []
    fold_bar = tqdm(cv.split(X, y), total=n_splits,
                     desc=f"    [{label}] CV folds", leave=True,
                     bar_format="{l_bar}{bar:25}{r_bar}")
    for train_idx, val_idx in fold_bar:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, random_state=SEED
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, proba)
        auc_scores.append(fold_auc)
        fold_bar.set_postfix({"fold_AUC": f"{fold_auc:.4f}"})
        del clf

    del X, y
    gc.collect()

    return float(np.mean(auc_scores)), float(np.std(auc_scores))


def main():
    results = load_results()
    print(f"Loaded {len(results)} existing results from {RESULTS_PATH}")
    print_progress_summary(results)

    X_all, y_all, proj_all = load_all_embeddings()

    n_projects = len(PROJECTS)
    for p_idx, project in enumerate(PROJECTS, start=1):
        print(f"\n=== Project {p_idx}/{n_projects}: {project} ===")

        if all(result_key(project, m) in results for m in METHODS):
            print(f"  all methods already computed -- skipping project.")
            continue

        try:
            source_emb, source_lbl, target_emb, target_lbl = build_project_split(
                X_all, y_all, proj_all, project
            )

            scaler = StandardScaler()
            source_emb_scaled = scaler.fit_transform(source_emb).astype(np.float32)
            target_emb_scaled = scaler.transform(target_emb).astype(np.float32)

            print(f"  Source: {source_emb_scaled.shape} (vuln={int(source_lbl.sum())}) "
                  f"| Target: {target_emb_scaled.shape} (vuln={int(target_lbl.sum())})")

            del source_emb, target_emb
            gc.collect()

        except Exception as e:
            print(f"  [FATAL for project {project}] Could not build split: {e}")
            traceback.print_exc()
            continue

        n_methods = len(METHODS)
        for m_idx, method in enumerate(METHODS, start=1):
            key = result_key(project, method)

            if key in results:
                print(f"  Method {m_idx}/{n_methods} [{method}]: already computed "
                      f"(AUC={results[key].get('mean_auc', 'N/A')}) -- skipping.")
                continue

            print(f"  Method {m_idx}/{n_methods} [{method}]:")

            t0 = time.time()
            resampled_source = None
            try:
                resampled_source = get_oversampled_source(
                    project, method, source_emb_scaled, source_lbl
                )

                mean_auc, std_auc = adversarial_validation(
                    resampled_source, target_emb_scaled, label=f"{project}/{method}"
                )

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
                print(f"    AUC: {mean_auc:.4f} +/- {std_auc:.4f} "
                      f"({time.time() - t0:.1f}s) -- saved.")
                print_progress_summary(results)

            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
                results[key] = {
                    "project": project,
                    "method": method,
                    "status": "failed",
                    "error": str(e),
                    "elapsed_sec": round(time.time() - t0, 1),
                }
                save_results(results)
                print_progress_summary(results)

            finally:
                if resampled_source is not None:
                    del resampled_source
                gc.collect()

        del source_emb_scaled, target_emb_scaled, source_lbl, target_lbl
        gc.collect()

    print("\nALL PROJECTS AND METHODS PROCESSED (OR ALREADY CACHED).")
    print_progress_summary(results)
    print(f"Final results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()