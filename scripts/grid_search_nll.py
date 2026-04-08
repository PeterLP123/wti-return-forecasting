"""
Grid Search Script: XGBoost (MSE) + LSTM (NLL) + LSTM (NLL+Dir+Var)
Saves results incrementally. Run from project root:
    python scripts/grid_search_nll.py
"""
import time
import json
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from itertools import product
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    print(msg, flush=True)


def _normalise_scalar(value):
    """Normalise scalar values so CSV-loaded params match in-memory grid params."""
    if pd.isna(value):
        return 'NaN'
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isfinite(value) and abs(float(value) - round(float(value))) < 1e-12:
            return str(int(round(float(value))))
        return format(float(value), '.12g')
    try:
        numeric = float(value)
        if np.isfinite(numeric) and abs(numeric - round(numeric)) < 1e-12:
            return str(int(round(numeric)))
        return format(numeric, '.12g')
    except (TypeError, ValueError):
        return str(value)


def _combo_key(params, keys):
    """Create a stable tuple key for one hyperparameter combination."""
    return tuple(_normalise_scalar(params[k]) for k in keys)

log(f'Device: {device}')

# ════════════════════════════════════════════════════════════════
# 1. DATA SETUP
# ════════════════════════════════════════════════════════════════

def find_project_root(start_path: Path) -> Path:
    """Locate repo root from this file path."""
    start_path = start_path.resolve()
    for candidate in [start_path] + list(start_path.parents):
        if (candidate / 'scripts' / 'grid_search_nll.py').exists() and (candidate / 'data' / 'processed').exists():
            return candidate
    raise FileNotFoundError(
        'Could not locate project root containing scripts/grid_search_nll.py and data/processed/.'
    )


PROJECT_DIR = find_project_root(Path(__file__).resolve().parent)
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
RESULTS_DIR = PROJECT_DIR / 'results' / 'nll_grid'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LSTM_ARTIFACT_FILENAMES = {
    'lstm_nll': {
        'params': 'lstm_nll_best_params.json',
        'model': 'lstm_nll_model.pth',
        'curves': 'lstm_nll_training_curves.csv',
    },
    'lstm_nll_dir_var': {
        'params': 'lstm_nll_dir_var_best_params.json',
        'model': 'lstm_nll_dir_var_model.pth',
        'curves': 'lstm_nll_dir_var_training_curves.csv',
    },
}


def get_lstm_artifact_paths(save_prefix: str) -> dict:
    """Return canonical artifact paths for probabilistic LSTM variants."""
    filenames = LSTM_ARTIFACT_FILENAMES.get(
        save_prefix,
        {
            'params': f'{save_prefix}_best_params.json',
            'model': f'{save_prefix}_model.pth',
            'curves': f'{save_prefix}_training_curves.csv',
        },
    )
    return {k: RESULTS_DIR / v for k, v in filenames.items()}

df = pd.read_parquet(PROCESSED_DIR / 'sanity_data_pca.parquet')
df['Date'] = pd.to_datetime(df['Date'])
TARGET = 'Target'
NON_FEATURE_COLS = ['Date', 'Target', 'WTI_Close', 'Brent_Close']
FEATURE_COLS = [c for c in df.columns if c not in NON_FEATURE_COLS]
df = df.dropna(subset=[TARGET] + FEATURE_COLS).reset_index(drop=True)

TRAIN_END = '2022-12-31'
VAL_END = '2024-06-30'

train_df = df[df['Date'] <= TRAIN_END]
val_df = df[(df['Date'] > TRAIN_END) & (df['Date'] <= VAL_END)]
test_df = df[df['Date'] > VAL_END]

scaler_X = RobustScaler()
scaler_y = RobustScaler()
train_X = scaler_X.fit_transform(train_df[FEATURE_COLS].values)
train_y = scaler_y.fit_transform(train_df[[TARGET]].values).flatten()
val_X = scaler_X.transform(val_df[FEATURE_COLS].values)
val_y = scaler_y.transform(val_df[[TARGET]].values).flatten()
test_X = scaler_X.transform(test_df[FEATURE_COLS].values)
test_y = scaler_y.transform(test_df[[TARGET]].values).flatten()
n_features = len(FEATURE_COLS)

log(f'Dataset: sanity_data_pca | Features: {n_features}')
log(f'Train: {len(train_X)} | Val: {len(val_X)} | Test: {len(test_X)}')

# ════════════════════════════════════════════════════════════════
# 2. HELPERS
# ════════════════════════════════════════════════════════════════
def create_sequences(features, target, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len + 1):
        X.append(features[i:i + seq_len])
        y.append(target[i + seq_len - 1])
    return np.array(X), np.array(y)


def create_sequences_with_bridge(prev_X, prev_y, curr_X, curr_y, sl):
    bX = np.vstack([prev_X[-(sl - 1):], curr_X])
    by = np.hstack([prev_y[-(sl - 1):], curr_y])
    return create_sequences(bX, by, sl)


class OilDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ════════════════════════════════════════════════════════════════
# 3. LSTM MODEL (NLL variant — outputs mu + log_var)
# ════════════════════════════════════════════════════════════════
class LSTMModelNLL(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(hidden_size, 1)
        self.fc_logvar = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc_mu(out), self.fc_logvar(out)


# ════════════════════════════════════════════════════════════════
# 4. LOSS FUNCTIONS
# ════════════════════════════════════════════════════════════════
def pure_nll(mu, log_var, target):
    """Gaussian negative log-likelihood."""
    var = torch.exp(log_var) + 1e-6
    return torch.mean(0.5 * (log_var + (target - mu) ** 2 / var))


def nll_dir_var(mu, log_var, target, beta=1.0, gamma=0.05, temp=10.0):
    """NLL + directional penalty + variance encouragement on mu."""
    var = torch.exp(log_var) + 1e-6
    nll = torch.mean(0.5 * (log_var + (target - mu) ** 2 / var))
    # Directional: penalise sign mismatches
    dir_correct = torch.sigmoid(temp * mu * target)
    dir_loss = 1.0 - torch.mean(dir_correct)
    # Variance: prevent mu from collapsing to constant
    mu_var = torch.var(mu)
    var_penalty = 1.0 / (mu_var + 1e-6)
    return nll + beta * dir_loss + gamma * var_penalty


# ════════════════════════════════════════════════════════════════
# 5. LSTM TRAINING
# ════════════════════════════════════════════════════════════════
EPOCHS = 50
BATCH_SIZE = 32


def train_lstm_fold(model, train_loader, val_loader, loss_fn, lr, patience=10):
    """Train one fold. Returns best val loss (pure NLL) and early stop epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    best_state = None
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0
        non_finite_train = False
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            mu, log_var = model(X_b)
            loss = loss_fn(mu, log_var, y_b)
            if not torch.isfinite(loss):
                non_finite_train = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
        if non_finite_train:
            log('  Warning: non-finite training loss encountered; stopping this fold early.')
            break
        train_losses.append(ep_loss / len(train_loader))

        model.eval()
        vl = 0
        non_finite_val = False
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                mu, log_var = model(X_b)
                batch_vl = pure_nll(mu, log_var, y_b).item()  # always validate on pure NLL
                if not np.isfinite(batch_vl):
                    non_finite_val = True
                    break
                vl += batch_vl
        if non_finite_val:
            avg_vl = float('inf')
            log('  Warning: non-finite validation loss encountered; marking fold loss as inf.')
        else:
            avg_vl = vl / len(val_loader)
        val_losses.append(avg_vl)

        if np.isfinite(avg_vl) and avg_vl < best_val:
            best_val = avg_vl
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        log('  Warning: no finite validation checkpoint captured; using current model weights.')
        best_state = deepcopy(model.state_dict())
        best_val = float('inf')
    model.load_state_dict(best_state)
    return model, best_val, epoch + 1, train_losses, val_losses


def evaluate_lstm(model, loader, scaler_y):
    """Evaluate LSTM NLL model on a loader. Returns metrics dict."""
    model.eval()
    mus, logvars, actuals = [], [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            mu, lv = model(X_b)
            mus.append(mu.cpu().numpy())
            logvars.append(lv.cpu().numpy())
            actuals.append(y_b.numpy())
    mus = np.concatenate(mus).flatten()
    logvars = np.concatenate(logvars).flatten()
    actuals = np.concatenate(actuals).flatten()
    preds = scaler_y.inverse_transform(mus.reshape(-1, 1)).flatten()
    acts = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
    sigma = np.sqrt(np.exp(logvars)) * scaler_y.scale_[0]
    rmse = np.sqrt(mean_squared_error(acts, preds))
    mae = mean_absolute_error(acts, preds)
    dir_acc = np.mean(np.sign(preds) == np.sign(acts))
    return {
        'RMSE': rmse, 'MAE': mae, 'Dir_Acc': dir_acc,
        'pred_std': preds.std(), 'avg_sigma': sigma.mean(),
        'preds': preds, 'actuals': acts, 'sigma': sigma,
    }


# ════════════════════════════════════════════════════════════════
# 6. LSTM GRID SEARCH RUNNER
# ════════════════════════════════════════════════════════════════
def run_lstm_grid(loss_fn, loss_name, grid, save_prefix):
    """Run LSTM grid search with given loss function."""
    log(f'\n{"="*70}')
    log(f'LSTM GRID SEARCH: {loss_name}')
    log(f'{"="*70}')

    keys = list(grid.keys())
    combos = list(product(*grid.values()))
    log(f'Configs: {len(combos)} | Folds: 3 | Max epochs: {EPOCHS}')

    results = []
    t0 = time.time()
    best_so_far = float('inf')
    csv_path = RESULTS_DIR / f'{save_prefix}_grid_results.csv'
    completed_keys = set()

    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty and all(k in existing_df.columns for k in keys):
                existing_by_key = {}
                for rec in existing_df.to_dict(orient='records'):
                    key = _combo_key(rec, keys)
                    prev = existing_by_key.get(key)
                    if prev is None:
                        existing_by_key[key] = rec
                        continue
                    prev_loss = float(prev.get('avg_cv_val_loss', float('inf')))
                    curr_loss = float(rec.get('avg_cv_val_loss', float('inf')))
                    if np.isfinite(curr_loss) and (not np.isfinite(prev_loss) or curr_loss < prev_loss):
                        existing_by_key[key] = rec
                results = list(existing_by_key.values())
                completed_keys = set(existing_by_key.keys())
                finite_losses = [
                    float(r['avg_cv_val_loss'])
                    for r in results
                    if pd.notna(r.get('avg_cv_val_loss')) and np.isfinite(float(r.get('avg_cv_val_loss')))
                ]
                if finite_losses:
                    best_so_far = min(finite_losses)
                log(f'Resuming {loss_name}: loaded {len(results)} completed configs from {csv_path}')
            else:
                log(f'Found existing CSV at {csv_path}, but required columns are missing; starting fresh.')
        except Exception as exc:
            log(f'Warning: could not load existing results from {csv_path}: {exc}')

    processed_this_run = 0

    try:
        for combo in combos:
            params = dict(zip(keys, combo))
            key = _combo_key(params, keys)
            if key in completed_keys:
                continue

            sl = params['seq_len']
            X_all, y_all = create_sequences(train_X, train_y, sl)
            tscv = TimeSeriesSplit(n_splits=3)
            fold_losses = []
            fold_epochs = []

            for tr_idx, vl_idx in tscv.split(X_all):
                tr_loader = DataLoader(OilDataset(X_all[tr_idx], y_all[tr_idx]),
                                       batch_size=BATCH_SIZE, shuffle=False)
                vl_loader = DataLoader(OilDataset(X_all[vl_idx], y_all[vl_idx]),
                                       batch_size=BATCH_SIZE, shuffle=False)
                torch.manual_seed(SEED)
                model = LSTMModelNLL(
                    n_features, params['hidden_size'],
                    params['num_layers'], params['dropout'],
                ).to(device)
                _, vl_loss, ep, _, _ = train_lstm_fold(
                    model, tr_loader, vl_loader, loss_fn, params['learning_rate'],
                )
                fold_losses.append(vl_loss)
                fold_epochs.append(ep)

            avg_loss = np.mean(fold_losses)
            rec = {
                **params,
                'avg_cv_val_loss': avg_loss,
                'fold1_loss': fold_losses[0],
                'fold2_loss': fold_losses[1],
                'fold3_loss': fold_losses[2],
                'fold1_epochs': fold_epochs[0],
                'fold2_epochs': fold_epochs[1],
                'fold3_epochs': fold_epochs[2],
                'avg_epochs': np.mean(fold_epochs),
            }
            results.append(rec)
            completed_keys.add(key)
            processed_this_run += 1

            is_best = np.isfinite(avg_loss) and avg_loss < best_so_far
            if is_best:
                best_so_far = avg_loss

            done_total = len(completed_keys)
            # Save incrementally every 50 new configs
            if (processed_this_run % 50 == 0 and processed_this_run > 0) or done_total == len(combos):
                pd.DataFrame(results).sort_values('avg_cv_val_loss').to_csv(csv_path, index=False)

            if (processed_this_run % 25 == 0 and processed_this_run > 0) or done_total == len(combos) or is_best:
                elapsed = time.time() - t0
                rate = processed_this_run / elapsed if elapsed > 0 else 0
                eta = (len(combos) - done_total) / rate if rate > 0 else 0
                marker = ' *** NEW BEST ***' if is_best else ''
                log(f'  [{done_total:5d}/{len(combos)}] loss={avg_loss:.6f} '
                    f'ep={np.mean(fold_epochs):.0f} | '
                    f'{elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m ({eta/3600:.1f}h){marker}')

    except KeyboardInterrupt:
        log(f'\n*** Interrupted after {len(completed_keys)}/{len(combos)} configs ***')

    finally:
        results_df = pd.DataFrame(results)
        if results:
            results_df = results_df.sort_values('avg_cv_val_loss').reset_index(drop=True)
            results_df.to_csv(csv_path, index=False)
            log(f'\nSaved {len(results)} results to {csv_path}')

    elapsed_total = time.time() - t0
    log(f'{loss_name} complete. {len(completed_keys)} configs in {elapsed_total/60:.1f}m ({elapsed_total/3600:.1f}h)')
    log(f'\nTop 5:')
    log(results_df.head().to_string(index=False))
    return results_df


# ════════════════════════════════════════════════════════════════
# 7. XGBOOST GRID SEARCH
# ════════════════════════════════════════════════════════════════
def run_xgb_grid(grid, save_prefix):
    """Run XGBoost grid search with MSE (= Gaussian NLL, fixed variance)."""
    log(f'\n{"="*70}')
    log(f'XGBOOST GRID SEARCH (MSE / Gaussian NLL, fixed variance)')
    log(f'{"="*70}')

    keys = list(grid.keys())
    combos = list(product(*grid.values()))
    log(f'Configs: {len(combos)} | Folds: 3')

    results = []
    t0 = time.time()
    best_so_far = float('inf')
    csv_path = RESULTS_DIR / f'{save_prefix}_grid_results.csv'
    completed_keys = set()

    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty and all(k in existing_df.columns for k in keys):
                existing_by_key = {}
                for rec in existing_df.to_dict(orient='records'):
                    key = _combo_key(rec, keys)
                    prev = existing_by_key.get(key)
                    if prev is None:
                        existing_by_key[key] = rec
                        continue
                    prev_loss = float(prev.get('avg_cv_val_loss', float('inf')))
                    curr_loss = float(rec.get('avg_cv_val_loss', float('inf')))
                    if np.isfinite(curr_loss) and (not np.isfinite(prev_loss) or curr_loss < prev_loss):
                        existing_by_key[key] = rec
                results = list(existing_by_key.values())
                completed_keys = set(existing_by_key.keys())
                finite_losses = [
                    float(r['avg_cv_val_loss'])
                    for r in results
                    if pd.notna(r.get('avg_cv_val_loss')) and np.isfinite(float(r.get('avg_cv_val_loss')))
                ]
                if finite_losses:
                    best_so_far = min(finite_losses)
                log(f'Resuming XGBoost: loaded {len(results)} completed configs from {csv_path}')
            else:
                log(f'Found existing CSV at {csv_path}, but required columns are missing; starting fresh.')
        except Exception as exc:
            log(f'Warning: could not load existing results from {csv_path}: {exc}')

    processed_this_run = 0

    try:
        for combo in combos:
            params = dict(zip(keys, combo))
            key = _combo_key(params, keys)
            if key in completed_keys:
                continue

            tscv = TimeSeriesSplit(n_splits=3)
            fold_losses = []

            for tr_idx, vl_idx in tscv.split(train_X):
                xgb = XGBRegressor(
                    **params,
                    objective='reg:squarederror',
                    random_state=SEED,
                    verbosity=0,
                )
                xgb.fit(
                    train_X[tr_idx], train_y[tr_idx],
                    eval_set=[(train_X[vl_idx], train_y[vl_idx])],
                    verbose=False,
                )
                preds = xgb.predict(train_X[vl_idx])
                fold_losses.append(mean_squared_error(train_y[vl_idx], preds))

            avg_loss = np.mean(fold_losses)
            rec = {
                **params,
                'avg_cv_val_loss': avg_loss,
                'fold1_loss': fold_losses[0],
                'fold2_loss': fold_losses[1],
                'fold3_loss': fold_losses[2],
            }
            results.append(rec)
            completed_keys.add(key)
            processed_this_run += 1

            is_best = np.isfinite(avg_loss) and avg_loss < best_so_far
            if is_best:
                best_so_far = avg_loss

            done_total = len(completed_keys)
            # Save incrementally every 100 new configs
            if (processed_this_run % 100 == 0 and processed_this_run > 0) or done_total == len(combos):
                pd.DataFrame(results).sort_values('avg_cv_val_loss').to_csv(csv_path, index=False)

            if (processed_this_run % 100 == 0 and processed_this_run > 0) or done_total == len(combos) or is_best:
                elapsed = time.time() - t0
                rate = processed_this_run / elapsed if elapsed > 0 else 0
                eta = (len(combos) - done_total) / rate if rate > 0 else 0
                marker = ' *** NEW BEST ***' if is_best else ''
                log(f'  [{done_total:5d}/{len(combos)}] loss={avg_loss:.6f} | '
                    f'{elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m ({eta/3600:.1f}h){marker}')

    except KeyboardInterrupt:
        log(f'\n*** Interrupted after {len(completed_keys)}/{len(combos)} configs ***')

    finally:
        results_df = pd.DataFrame(results)
        if results:
            results_df = results_df.sort_values('avg_cv_val_loss').reset_index(drop=True)
            results_df.to_csv(csv_path, index=False)
            log(f'\nSaved {len(results)} results to {csv_path}')

    elapsed_total = time.time() - t0
    log(f'XGBoost complete. {len(completed_keys)} configs in {elapsed_total/60:.1f}m ({elapsed_total/3600:.1f}h)')
    log(f'\nTop 5:')
    log(results_df.head().to_string(index=False))
    return results_df


# ════════════════════════════════════════════════════════════════
# 8. GRID DEFINITIONS
# ════════════════════════════════════════════════════════════════
LSTM_GRID = {
    'seq_len':       [3, 5, 7, 10, 15, 20, 30, 40, 60],
    'hidden_size':   [16, 32, 48, 64, 96, 128],
    'num_layers':    [1, 2, 3],
    'dropout':       [0.0, 0.1, 0.3, 0.5],
    'learning_rate': [1e-3, 5e-4, 1e-4],
}
# 9 * 6 * 3 * 4 * 3 = 1944 configs

XGB_GRID = {
    'n_estimators':     [50, 100, 150, 200, 300, 500],
    'max_depth':        [2, 3, 4, 5, 7],
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8],
    'reg_alpha':        [0, 0.1],
    'reg_lambda':       [0.5, 1.0, 2.0],
}
# 6 * 5 * 3 * 3 * 2 * 2 * 3 = 3240 configs

n_lstm = 1
for v in LSTM_GRID.values():
    n_lstm *= len(v)
n_xgb = 1
for v in XGB_GRID.values():
    n_xgb *= len(v)

log(f'\nGrid sizes: LSTM={n_lstm} (x2 rounds) | XGBoost={n_xgb}')
log(f'Estimated time: LSTM ~{n_lstm*2*4.5/3600:.1f}h | XGBoost ~{n_xgb*1.8/3600:.1f}h')
log(f'Total estimated: ~{(n_lstm*2*4.5 + n_xgb*1.8)/3600:.1f}h')


# ════════════════════════════════════════════════════════════════
# 9. RUN ALL GRID SEARCHES
# ════════════════════════════════════════════════════════════════
t_global = time.time()

# --- Grid Search 1: XGBoost ---
xgb_results = run_xgb_grid(XGB_GRID, 'xgb')

# --- Grid Search 2: LSTM Pure NLL ---
lstm_nll_results = run_lstm_grid(pure_nll, 'Pure NLL', LSTM_GRID, 'lstm_nll')

# --- Grid Search 3: LSTM NLL+Dir+Var ---
lstm_combo_results = run_lstm_grid(
    lambda mu, lv, t: nll_dir_var(mu, lv, t, beta=1.0, gamma=0.05),
    'NLL + Dir + Var (β=1.0, γ=0.05)',
    LSTM_GRID,
    'lstm_nll_dir_var',
)


# ════════════════════════════════════════════════════════════════
# 10. TRAIN FINAL MODELS WITH BEST CONFIGS
# ════════════════════════════════════════════════════════════════
log(f'\n{"="*70}')
log(f'TRAINING FINAL MODELS WITH BEST CONFIGS')
log(f'{"="*70}')


def train_final_lstm(results_df, loss_fn, loss_name, save_prefix):
    """Train final LSTM model with best config, save curves and test metrics."""
    best = results_df.iloc[0]
    sl = int(best['seq_len'])
    hidden = int(best['hidden_size'])
    layers = int(best['num_layers'])
    dropout = float(best['dropout'])
    lr = float(best['learning_rate'])

    log(f'\n{loss_name} best: sl={sl} h={hidden} L={layers} d={dropout} lr={lr}')

    X_tr, y_tr = create_sequences(train_X, train_y, sl)
    X_vl, y_vl = create_sequences_with_bridge(train_X, train_y, val_X, val_y, sl)
    X_te, y_te = create_sequences_with_bridge(val_X, val_y, test_X, test_y, sl)

    tr_loader = DataLoader(OilDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=False)
    vl_loader = DataLoader(OilDataset(X_vl, y_vl), batch_size=BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(OilDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(SEED)
    model = LSTMModelNLL(n_features, hidden, layers, dropout).to(device)
    model, _, ep, train_losses, val_losses = train_lstm_fold(
        model, tr_loader, vl_loader, loss_fn, lr, patience=15,
    )
    log(f'  Trained {ep} epochs')

    artifact_paths = get_lstm_artifact_paths(save_prefix)

    # Save training curves
    curves = pd.DataFrame({'epoch': range(1, len(train_losses)+1),
                           'train_loss': train_losses, 'val_loss': val_losses})
    curves.to_csv(artifact_paths['curves'], index=False)

    # Test evaluation
    test_metrics = evaluate_lstm(model, te_loader, scaler_y)
    log(f'  Test RMSE={test_metrics["RMSE"]:.6f} MAE={test_metrics["MAE"]:.6f} '
        f'Dir={test_metrics["Dir_Acc"]:.4f} PredStd={test_metrics["pred_std"]:.6f} '
        f'Sigma={test_metrics["avg_sigma"]:.6f}')

    # Save best params
    best_params = {
        'seq_len': sl, 'hidden_size': hidden, 'num_layers': layers,
        'dropout': dropout, 'learning_rate': lr, 'loss': loss_name,
        'epochs_trained': ep,
    }
    with open(artifact_paths['params'], 'w') as f:
        json.dump(best_params, f, indent=2)

    # Save model
    torch.save(model.state_dict(), artifact_paths['model'])
    log(f'  Saved checkpoint: {artifact_paths["model"]}')
    log(f'  Saved params: {artifact_paths["params"]}')
    log(f'  Saved curves: {artifact_paths["curves"]}')

    return test_metrics, best_params


def train_final_xgb(results_df, save_prefix):
    """Train final XGBoost with best config, evaluate on test."""
    best = results_df.iloc[0]
    params = best.drop([c for c in best.index if 'loss' in c or 'fold' in c]).to_dict()
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])

    log(f'\nXGBoost best: {params}')

    xgb_model = XGBRegressor(
        **params, objective='reg:squarederror',
        random_state=SEED, verbosity=0, early_stopping_rounds=20,
    )
    xgb_model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
    log(f'  Best iteration: {xgb_model.best_iteration}')

    # Test predictions
    xgb_preds_s = xgb_model.predict(test_X)
    xgb_preds = scaler_y.inverse_transform(xgb_preds_s.reshape(-1, 1)).flatten()
    xgb_acts = scaler_y.inverse_transform(test_y.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(xgb_acts, xgb_preds))
    mae = mean_absolute_error(xgb_acts, xgb_preds)
    dir_acc = np.mean(np.sign(xgb_preds) == np.sign(xgb_acts))
    pred_std = xgb_preds.std()

    log(f'  Test RMSE={rmse:.6f} MAE={mae:.6f} Dir={dir_acc:.4f} PredStd={pred_std:.6f}')

    # Save
    best_params = {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                   for k, v in params.items()}
    best_params['loss'] = 'MSE (Gaussian NLL, fixed variance)'
    best_params['best_iteration'] = int(xgb_model.best_iteration)
    with open(RESULTS_DIR / f'{save_prefix}_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    xgb_model.save_model(str(RESULTS_DIR / f'{save_prefix}_model.json'))

    return {
        'RMSE': rmse, 'MAE': mae, 'Dir_Acc': dir_acc,
        'pred_std': pred_std, 'preds': xgb_preds, 'actuals': xgb_acts,
    }, best_params


# Train finals
xgb_metrics, xgb_params = train_final_xgb(xgb_results, 'xgb')
nll_metrics, nll_params = train_final_lstm(lstm_nll_results, pure_nll, 'Pure NLL', 'lstm_nll')
combo_metrics, combo_params = train_final_lstm(
    lstm_combo_results,
    lambda mu, lv, t: nll_dir_var(mu, lv, t, beta=1.0, gamma=0.05),
    'NLL+Dir+Var', 'lstm_nll_dir_var',
)


# ════════════════════════════════════════════════════════════════
# 11. AR(10) BASELINE
# ════════════════════════════════════════════════════════════════
log(f'\n{"="*70}')
log(f'AR(10) BASELINE')
log(f'{"="*70}')

ar_order = 10
target_vals = df[TARGET].values
ar_X_all = np.column_stack([np.roll(target_vals, i) for i in range(1, ar_order + 1)])
ar_y_all = target_vals.copy()
valid = np.arange(ar_order, len(target_vals))
ar_X_all, ar_y_all = ar_X_all[valid], ar_y_all[valid]
ar_dates = df['Date'].values[valid]

train_mask = ar_dates <= np.datetime64(TRAIN_END)
test_mask = ar_dates > np.datetime64(VAL_END)
ar_X_tr, ar_y_tr = ar_X_all[train_mask], ar_y_all[train_mask]
ar_X_te, ar_y_te = ar_X_all[test_mask], ar_y_all[test_mask]

beta_ar = np.linalg.lstsq(
    np.column_stack([np.ones(len(ar_X_tr)), ar_X_tr]), ar_y_tr, rcond=None
)[0]
ar_preds = np.column_stack([np.ones(len(ar_X_te)), ar_X_te]) @ beta_ar

ar_rmse = np.sqrt(mean_squared_error(ar_y_te, ar_preds))
ar_mae = mean_absolute_error(ar_y_te, ar_preds)
ar_dir = np.mean(np.sign(ar_preds) == np.sign(ar_y_te))
log(f'AR(10): RMSE={ar_rmse:.6f} MAE={ar_mae:.6f} Dir={ar_dir:.4f}')


# ════════════════════════════════════════════════════════════════
# 12. FINAL COMPARISON
# ════════════════════════════════════════════════════════════════
# Trim XGBoost to match LSTM test window
lstm_n = len(nll_metrics['preds'])
xgb_preds_t = xgb_metrics['preds'][-lstm_n:]
xgb_acts_t = xgb_metrics['actuals'][-lstm_n:]
xgb_rmse_t = np.sqrt(mean_squared_error(xgb_acts_t, xgb_preds_t))
xgb_mae_t = mean_absolute_error(xgb_acts_t, xgb_preds_t)
xgb_dir_t = np.mean(np.sign(xgb_preds_t) == np.sign(xgb_acts_t))

naive_rmse = np.sqrt(mean_squared_error(nll_metrics['actuals'], np.zeros(lstm_n)))
naive_mae = mean_absolute_error(nll_metrics['actuals'], np.zeros(lstm_n))

log(f'\n{"="*70}')
log(f'FINAL COMPARISON (all on same test set)')
log(f'{"="*70}')
log(f'{"Model":<30} {"Loss":<25} {"RMSE":<10} {"MAE":<10} {"Dir Acc":<10} {"Pred Std":<10}')
log(f'{"-"*95}')
log(f'{"LSTM":<30} {"Pure NLL":<25} {nll_metrics["RMSE"]:<10.6f} {nll_metrics["MAE"]:<10.6f} '
    f'{nll_metrics["Dir_Acc"]:<10.4f} {nll_metrics["pred_std"]:<10.6f}')
log(f'{"LSTM":<30} {"NLL+Dir+Var":<25} {combo_metrics["RMSE"]:<10.6f} {combo_metrics["MAE"]:<10.6f} '
    f'{combo_metrics["Dir_Acc"]:<10.4f} {combo_metrics["pred_std"]:<10.6f}')
log(f'{"XGBoost":<30} {"MSE (Gaussian NLL)":<25} {xgb_rmse_t:<10.6f} {xgb_mae_t:<10.6f} '
    f'{xgb_dir_t:<10.4f} {xgb_preds_t.std():<10.6f}')
log(f'{"AR(10)":<30} {"OLS (Gaussian MLE)":<25} {ar_rmse:<10.6f} {ar_mae:<10.6f} '
    f'{ar_dir:<10.4f} {ar_preds.std():<10.6f}')
log(f'{"Naive (predict 0)":<30} {"—":<25} {naive_rmse:<10.6f} {naive_mae:<10.6f} '
    f'{"0.5000":<10} {"0.0000":<10}')

# Save comparison
comp = pd.DataFrame([
    {'Model': 'LSTM (Pure NLL)', 'RMSE': nll_metrics['RMSE'], 'MAE': nll_metrics['MAE'],
     'Dir_Acc': nll_metrics['Dir_Acc'], 'Pred_Std': nll_metrics['pred_std'],
     'Avg_Sigma': nll_metrics['avg_sigma']},
    {'Model': 'LSTM (NLL+Dir+Var)', 'RMSE': combo_metrics['RMSE'], 'MAE': combo_metrics['MAE'],
     'Dir_Acc': combo_metrics['Dir_Acc'], 'Pred_Std': combo_metrics['pred_std'],
     'Avg_Sigma': combo_metrics['avg_sigma']},
    {'Model': 'XGBoost', 'RMSE': xgb_rmse_t, 'MAE': xgb_mae_t,
     'Dir_Acc': xgb_dir_t, 'Pred_Std': xgb_preds_t.std(), 'Avg_Sigma': np.nan},
    {'Model': 'AR(10)', 'RMSE': ar_rmse, 'MAE': ar_mae,
     'Dir_Acc': ar_dir, 'Pred_Std': ar_preds.std(), 'Avg_Sigma': np.nan},
    {'Model': 'Naive', 'RMSE': naive_rmse, 'MAE': naive_mae,
     'Dir_Acc': 0.5, 'Pred_Std': 0.0, 'Avg_Sigma': np.nan},
])
comp.to_csv(RESULTS_DIR / 'final_comparison.csv', index=False)

elapsed_global = time.time() - t_global
log(f'\n{"="*70}')
log(f'ALL DONE. Total time: {elapsed_global/60:.1f}m ({elapsed_global/3600:.1f}h)')
log(f'Results saved to: {RESULTS_DIR}')
log(f'{"="*70}')
