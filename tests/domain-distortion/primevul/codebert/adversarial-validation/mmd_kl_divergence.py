"""
MMD^2 and KL Divergence across PrimeVul (Scenarios 1-4)
------------------------------------------------------------
"""

import os
import gc
import json
import time
import traceback
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore')


COMBINED_FILE = "../../../../../embedding/primevul/codebert/primevul_embedded.jsonl"
EMB_KEY       = "emb"
CACHE_DIR     = "results/oversample_cache"
RESULTS_PATH  = "results/mmd_kl_primevul.json"
SEED          = 42

PCA_COMPONENTS = 50
KL_BINS        = 50
MMD_GAMMA      = 1.0

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
np.random.seed(SEED)

PROJECTS = [
    "linux", "Chrome", "qemu", "gpac",
    "poppler", "radare2_s2", "linux-2.6", "vim", "FFmpeg",
    "php-src", "Android", "openssl", "ImageMagick", "tensorflow",
    "tcpdump", "radare2_s4", "FreeRDP",
]

METHODS = ["No Oversampling", "SMOTE", "ADASYN", "SVMSMOTE", "GAN"]

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


def load_source_for_method(project, method, source_emb_scaled):
    if method == "No Oversampling":
        return source_emb_scaled

    cache_path = os.path.join(CACHE_DIR, f"{project}_{method}.npy")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cached {method} embeddings not found for {project} at {cache_path}. "
            f"Run the corresponding oversampling/GAN script first."
        )
    return np.load(cache_path)


def compute_mmd(X, Y, gamma=MMD_GAMMA):
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def compute_avg_kl(source_pca, target_pca, bins=KL_BINS):
    kl_values = []
    for i in range(source_pca.shape[1]):
        hist_source, edges = np.histogram(source_pca[:, i], bins=bins, density=True)
        hist_target, _ = np.histogram(target_pca[:, i], bins=edges, density=True)

        hist_source = hist_source + 1e-10
        hist_target = hist_target + 1e-10

        kl = entropy(hist_target, hist_source)
        kl_values.append(kl)

    return float(np.mean(kl_values))


def run_project_method(project, method, source_emb_scaled, target_emb_scaled):
    source_for_mmd = load_source_for_method(project, method, source_emb_scaled)

    n_source = min(source_for_mmd.shape[0], 20000)
    n_target = min(target_emb_scaled.shape[0], 20000)

    rng = np.random.RandomState(SEED)
    source_sample = source_for_mmd[rng.choice(source_for_mmd.shape[0], n_source, replace=False)] \
        if source_for_mmd.shape[0] > n_source else source_for_mmd
    target_sample = target_emb_scaled[rng.choice(target_emb_scaled.shape[0], n_target, replace=False)] \
        if target_emb_scaled.shape[0] > n_target else target_emb_scaled

    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
    pca.fit(np.vstack([source_sample, target_sample]))

    source_pca = pca.transform(source_sample)
    target_pca = pca.transform(target_sample)

    mmd = compute_mmd(source_pca, target_pca)
    avg_kl = compute_avg_kl(source_pca, target_pca)

    return {
        "mmd_squared": round(mmd, 6),
        "avg_kl_divergence": round(avg_kl, 6),
        "n_source_used": int(source_sample.shape[0]),
        "n_target_used": int(target_sample.shape[0]),
        "n_source_full": int(source_for_mmd.shape[0]),
        "n_target_full": int(target_emb_scaled.shape[0]),
    }


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

            print(f"  Source: {source_emb_scaled.shape} | Target: {target_emb_scaled.shape}")

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
                      f"(MMD^2={results[key].get('mmd_squared', 'N/A')}) -- skipping.")
                continue

            print(f"  Method {m_idx}/{n_methods} [{method}]:")

            t0 = time.time()
            try:
                metrics = run_project_method(project, method, source_emb_scaled, target_emb_scaled)
                metrics["project"] = project
                metrics["method"] = method
                metrics["elapsed_sec"] = round(time.time() - t0, 1)
                metrics["status"] = "ok"
                results[key] = metrics
                save_results(results)
                print(f"    MMD^2={metrics['mmd_squared']:.6f} | "
                      f"Avg KL={metrics['avg_kl_divergence']:.6f} "
                      f"({metrics['elapsed_sec']}s) -- saved.")
                print_progress_summary(results)

            except FileNotFoundError as e:
                print(f"    SKIPPED (cache not ready): {e}")
                continue

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
                gc.collect()

        del source_emb_scaled, target_emb_scaled, source_lbl, target_lbl
        gc.collect()

    print("\nALL PROJECTS AND METHODS PROCESSED (OR ALREADY CACHED).")
    print_progress_summary(results)
    print(f"Final results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()