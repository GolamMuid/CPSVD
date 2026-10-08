"""
Adversarial Validation across PrimeVul -- GAN Oversampling Only
------------------------------------------------------------------
"""

import os
import gc
import json
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


COMBINED_FILE = "../../../../../embedding/primevul/codebert/primevul_embedded.jsonl"
EMB_KEY       = "emb"
RESULTS_PATH  = "results/adversarial_validation_primevul.json"
CACHE_DIR     = "results/oversample_cache"
SEED          = 42
N_SPLITS      = 5

GAN_LATENT_DIM  = 100
GAN_EPOCHS      = 1500
GAN_BATCH_SIZE  = 64
GAN_LOG_EVERY   = 500

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = "mps"
    torch.mps.manual_seed(SEED)
elif torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.manual_seed_all(SEED)
else:
    DEVICE = "cpu"

PROJECTS = [
    "linux", "Chrome", "qemu", "gpac",
    "poppler", "radare2_s2", "linux-2.6", "vim", "FFmpeg",
    "php-src", "Android", "openssl", "ImageMagick", "tensorflow",
    "tcpdump", "radare2_s4", "FreeRDP",
]

METHOD = "GAN"
TOTAL_RUNS = len(PROJECTS)


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


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
    done = sum(1 for k, v in results.items() if k.endswith(f"__{METHOD}") and v.get("status") == "ok")
    failed = sum(1 for k, v in results.items() if k.endswith(f"__{METHOD}") and v.get("status") == "failed")
    remaining = TOTAL_RUNS - done - failed
    pct = 100 * (done + failed) / TOTAL_RUNS
    print(f"----- GAN PROGRESS: {done} ok | {failed} failed | {remaining} remaining "
          f"| {pct:.1f}% of {TOTAL_RUNS} projects -----")


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
          f"({X.nbytes / 1e9:.2f} GB) | Device: {DEVICE}")
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


def train_gan_and_generate(project, source_emb, source_lbl):
    gan_cache_path = os.path.join(CACHE_DIR, f"{project}_GAN.npy")
    if os.path.exists(gan_cache_path):
        return np.load(gan_cache_path)

    input_dim = source_emb.shape[1]
    minority_emb = source_emb[source_lbl == 1].astype(np.float32)
    n_to_generate = int(np.sum(source_lbl == 0)) - int(np.sum(source_lbl == 1))

    print(f"    Minority samples: {minority_emb.shape[0]} | Target to generate: {n_to_generate}")

    G = Generator(GAN_LATENT_DIM, input_dim).to(DEVICE)
    D = Discriminator(input_dim).to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    minority_tensor = torch.tensor(minority_emb).to(DEVICE)
    dataset = TensorDataset(minority_tensor)
    loader = DataLoader(dataset, batch_size=GAN_BATCH_SIZE, shuffle=True)

    t0 = time.time()
    epoch_bar = tqdm(range(GAN_EPOCHS), desc=f"    [{project}] GAN training", leave=False)
    for epoch in epoch_bar:
        for (real_batch,) in loader:
            bs = real_batch.size(0)

            z = torch.randn(bs, GAN_LATENT_DIM).to(DEVICE)
            fake = G(z).detach()
            loss_D = criterion(D(real_batch), torch.ones(bs, 1).to(DEVICE)) + \
                     criterion(D(fake), torch.zeros(bs, 1).to(DEVICE))
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            z = torch.randn(bs, GAN_LATENT_DIM).to(DEVICE)
            loss_G = criterion(D(G(z)), torch.ones(bs, 1).to(DEVICE))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        if (epoch + 1) % GAN_LOG_EVERY == 0:
            epoch_bar.set_postfix({"D_loss": f"{loss_D.item():.4f}", "G_loss": f"{loss_G.item():.4f}"})

    print(f"    GAN training complete in {time.time() - t0:.1f}s")

    G.eval()
    with torch.no_grad():
        z = torch.randn(n_to_generate, GAN_LATENT_DIM).to(DEVICE)
        synthetic = G(z).cpu().numpy()

    resampled_emb = np.vstack([source_emb, synthetic]).astype(np.float32)
    np.save(gan_cache_path, resampled_emb)

    del G, D, opt_G, opt_D, minority_tensor, dataset, loader
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

    return resampled_emb


def adversarial_validation(source_emb, target_emb, label="", n_splits=N_SPLITS):
    X = np.vstack([source_emb, target_emb])
    y = np.array([0] * len(source_emb) + [1] * len(target_emb))

    print(f"    Fitting on {X.shape[0]} rows...")

    clf = HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=SEED)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    auc_scores = []
    fold_bar = tqdm(cv.split(X, y), total=n_splits,
                     desc=f"    [{label}] CV folds", leave=True,
                     bar_format="{l_bar}{bar:25}{r_bar}")
    for train_idx, val_idx in fold_bar:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, proba)
        auc_scores.append(fold_auc)
        fold_bar.set_postfix({"fold_AUC": f"{fold_auc:.4f}"})

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
        key = result_key(project, METHOD)
        print(f"\n=== Project {p_idx}/{n_projects}: {project} [{METHOD}] ===")

        if key in results and results[key].get("status") == "ok":
            print(f"  already computed (AUC={results[key].get('mean_auc', 'N/A')}) -- skipping.")
            continue

        t0 = time.time()
        resampled_source = None
        try:
            source_emb, source_lbl, target_emb, target_lbl = build_project_split(
                X_all, y_all, proj_all, project
            )

            scaler = StandardScaler()
            source_emb_scaled = scaler.fit_transform(source_emb).astype(np.float32)
            target_emb_scaled = scaler.transform(target_emb).astype(np.float32)

            print(f"  Source: {source_emb_scaled.shape} (vuln={int(source_lbl.sum())}) "
                  f"| Target: {target_emb_scaled.shape} (vuln={int(target_lbl.sum())})")

            resampled_source = train_gan_and_generate(project, source_emb_scaled, source_lbl)

            mean_auc, std_auc = adversarial_validation(
                resampled_source, target_emb_scaled, label=f"{project}/{METHOD}"
            )

            results[key] = {
                "project": project,
                "method": METHOD,
                "n_source": int(resampled_source.shape[0]),
                "n_target": int(target_emb_scaled.shape[0]),
                "mean_auc": round(mean_auc, 4),
                "std_auc": round(std_auc, 4),
                "elapsed_sec": round(time.time() - t0, 1),
                "status": "ok",
            }
            save_results(results)
            print(f"  AUC: {mean_auc:.4f} +/- {std_auc:.4f} "
                  f"({time.time() - t0:.1f}s) -- saved.")
            print_progress_summary(results)

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[key] = {
                "project": project,
                "method": METHOD,
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

    print("\nALL PROJECTS PROCESSED FOR GAN (OR ALREADY CACHED).")
    print_progress_summary(results)
    print(f"Final results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()