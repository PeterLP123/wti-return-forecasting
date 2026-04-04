"""Split plots into NLL-only set and custom loss set."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import sys
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#333333', 'axes.labelcolor': '#333333',
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.labelsize': 10,
    'xtick.color': '#333333', 'ytick.color': '#333333',
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'legend.framealpha': 0.9, 'legend.edgecolor': '#cccccc',
    'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'grid.alpha': 0.7,
    'font.family': 'serif', 'font.size': 10, 'figure.dpi': 150,
})

C_ACTUAL = '#2c3e50'
C_NLL    = '#c0392b'
C_COMBO  = '#8e44ad'
C_XGB    = '#27ae60'
C_AR     = '#d4a017'
C_DIR    = '#2980b9'

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

PROJECT_DIR = Path('.')
FINAL_DIR = PROJECT_DIR / 'results' / 'final'

# ── Data ──
df = pd.read_parquet('data/processed/sanity_data_pca.parquet')
df['Date'] = pd.to_datetime(df['Date'])
TARGET = 'Target'
NON_FEATURE_COLS = ['Date', 'Target', 'WTI_Close', 'Brent_Close']
FEATURE_COLS = [c for c in df.columns if c not in NON_FEATURE_COLS]
df = df.dropna(subset=[TARGET] + FEATURE_COLS).reset_index(drop=True)

train_df = df[df['Date'] <= '2022-12-31']
val_df = df[(df['Date'] > '2022-12-31') & (df['Date'] <= '2024-06-30')]
test_df = df[df['Date'] > '2024-06-30']

scaler_X = RobustScaler()
scaler_y = RobustScaler()
train_X = scaler_X.fit_transform(train_df[FEATURE_COLS].values)
train_y = scaler_y.fit_transform(train_df[[TARGET]].values).flatten()
val_X = scaler_X.transform(val_df[FEATURE_COLS].values)
val_y = scaler_y.transform(val_df[[TARGET]].values).flatten()
test_X = scaler_X.transform(test_df[FEATURE_COLS].values)
test_y = scaler_y.transform(test_df[[TARGET]].values).flatten()
n_features = len(FEATURE_COLS)

def create_sequences(features, target, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len + 1):
        X.append(features[i:i + seq_len])
        y.append(target[i + seq_len - 1])
    return np.array(X), np.array(y)

def create_sequences_with_bridge(pX, py, cX, cy, sl):
    bX = np.vstack([pX[-(sl-1):], cX])
    by = np.hstack([py[-(sl-1):], cy])
    return create_sequences(bX, by, sl)

class OilDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class LSTMModelNLL(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(hidden_size, 1)
        self.fc_logvar = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc_mu(out), self.fc_logvar(out)

def pure_nll(mu, log_var, target):
    var = torch.exp(log_var) + 1e-6
    return torch.mean(0.5 * (log_var + (target - mu) ** 2 / var))

def nll_dir_var(mu, log_var, target, beta=1.0, gamma=0.05, temp=10.0):
    var = torch.exp(log_var) + 1e-6
    nll = torch.mean(0.5 * (log_var + (target - mu) ** 2 / var))
    dir_correct = torch.sigmoid(temp * mu * target)
    dir_loss = 1.0 - torch.mean(dir_correct)
    mu_var = torch.var(mu)
    var_penalty = 1.0 / (mu_var + 1e-6)
    return nll + beta * dir_loss + gamma * var_penalty

def nll_dir_only(mu, log_var, target, beta=2.0, temp=10.0):
    var = torch.exp(log_var) + 1e-6
    nll = torch.mean(0.5 * (log_var + (target - mu) ** 2 / var))
    dir_correct = torch.sigmoid(temp * mu * target)
    dir_loss = 1.0 - torch.mean(dir_correct)
    return nll + beta * dir_loss

def train_lstm(sl, hidden, layers, dropout, lr, loss_fn, label):
    Xtr, ytr = create_sequences(train_X, train_y, sl)
    Xvl, yvl = create_sequences_with_bridge(train_X, train_y, val_X, val_y, sl)
    Xte, yte = create_sequences_with_bridge(val_X, val_y, test_X, test_y, sl)
    trl = DataLoader(OilDataset(Xtr, ytr), batch_size=32, shuffle=False)
    vll = DataLoader(OilDataset(Xvl, yvl), batch_size=32, shuffle=False)
    tel = DataLoader(OilDataset(Xte, yte), batch_size=32, shuffle=False)
    torch.manual_seed(SEED)
    model = LSTMModelNLL(n_features, hidden, layers, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bv, bs, w = float('inf'), None, 0
    tl_list, vl_list = [], []
    for epoch in range(60):
        model.train()
        ep = 0
        for Xb, yb in trl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(); mu, lv = model(Xb)
            loss = loss_fn(mu, lv, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ep += loss.item()
        tl_list.append(ep / len(trl))
        model.eval()
        vl = 0
        with torch.no_grad():
            for Xb, yb in vll:
                Xb, yb = Xb.to(device), yb.to(device)
                mu, lv = model(Xb); vl += pure_nll(mu, lv, yb).item()
        av = vl / len(vll); vl_list.append(av)
        if av < bv:
            bv = av; bs = {k: v.clone() for k, v in model.state_dict().items()}; w = 0
        else:
            w += 1
            if w >= 15: break
    model.load_state_dict(bs); model.eval()
    ms, lvs, acs = [], [], []
    with torch.no_grad():
        for Xb, yb in tel:
            Xb = Xb.to(device); mu, lv = model(Xb)
            ms.append(mu.cpu().numpy()); lvs.append(lv.cpu().numpy()); acs.append(yb.numpy())
    ms = np.concatenate(ms).flatten()
    lvs = np.concatenate(lvs).flatten()
    acs = np.concatenate(acs).flatten()
    preds = scaler_y.inverse_transform(ms.reshape(-1, 1)).flatten()
    acts = scaler_y.inverse_transform(acs.reshape(-1, 1)).flatten()
    sigma = np.sqrt(np.exp(lvs)) * scaler_y.scale_[0]
    rmse = np.sqrt(mean_squared_error(acts, preds))
    da = np.mean(np.sign(preds) == np.sign(acts))
    print(f'{label}: RMSE={rmse:.6f} Dir={da:.4f} PredStd={preds.std():.6f}')
    return preds, acts, sigma, tl_list, vl_list

def format_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

# ── Train all models ──
print('Training models...')
preds_nll, acts, sigma_nll, tl_nll, vl_nll = train_lstm(3, 128, 3, 0.0, 1e-3, pure_nll, 'Pure NLL')
preds_dir, _, sigma_dir, _, _ = train_lstm(5, 128, 2, 0.3, 5e-4, nll_dir_only, 'NLL+Dir')
preds_combo, _, sigma_combo, tl_combo, vl_combo = train_lstm(
    5, 128, 2, 0.3, 5e-4,
    lambda mu, lv, t: nll_dir_var(mu, lv, t, beta=1.0, gamma=0.05), 'NLL+Dir+Var')

xgb_model = XGBRegressor(n_estimators=50, max_depth=7, learning_rate=0.01,
                          subsample=0.9, colsample_bytree=0.7, reg_alpha=0.0,
                          reg_lambda=2.0, objective='reg:squarederror',
                          random_state=SEED, verbosity=0, early_stopping_rounds=20)
xgb_model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
xgb_preds = scaler_y.inverse_transform(xgb_model.predict(test_X).reshape(-1, 1)).flatten()[-len(acts):]
print(f'XGBoost: RMSE={np.sqrt(mean_squared_error(acts, xgb_preds)):.6f} '
      f'Dir={np.mean(np.sign(xgb_preds)==np.sign(acts)):.4f}')

ar_order = 10
target_vals = df[TARGET].values
ar_X_all = np.column_stack([np.roll(target_vals, i) for i in range(1, ar_order + 1)])
ar_y_all = target_vals.copy()
valid_idx = np.arange(ar_order, len(target_vals))
ar_X_all, ar_y_all = ar_X_all[valid_idx], ar_y_all[valid_idx]
ar_dates_all = df['Date'].values[valid_idx]
ar_X_tr = ar_X_all[ar_dates_all <= np.datetime64('2022-12-31')]
ar_y_tr = ar_y_all[ar_dates_all <= np.datetime64('2022-12-31')]
ar_X_te = ar_X_all[ar_dates_all > np.datetime64('2024-06-30')]
ar_y_te = ar_y_all[ar_dates_all > np.datetime64('2024-06-30')]
ar_test_dates = ar_dates_all[ar_dates_all > np.datetime64('2024-06-30')]
beta_ar = np.linalg.lstsq(np.column_stack([np.ones(len(ar_X_tr)), ar_X_tr]), ar_y_tr, rcond=None)[0]
ar_preds = np.column_stack([np.ones(len(ar_X_te)), ar_X_te]) @ beta_ar

test_dates = test_df['Date'].values[-len(acts):]

print('\n--- Generating NLL-only plots (06a-06d) ---')

# ══════════════════════════════════════════════════════════════
# 06a: NLL individual predictions — LSTM, XGBoost, AR(10) each vs actual
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
models = [
    (test_dates, acts, preds_nll, sigma_nll, 'LSTM (NLL)', C_NLL),
    (test_dates, acts, xgb_preds, None, 'XGBoost (MSE)', C_XGB),
    (ar_test_dates, ar_y_te, ar_preds, None, 'AR(10)', C_AR),
]
for i, (d, a, p, sig, label, color) in enumerate(models):
    ax = axes[i]
    rmse = np.sqrt(mean_squared_error(a, p))
    da = np.mean(np.sign(p) == np.sign(a))
    ax.plot(d, a, label='Actual', alpha=0.5, linewidth=0.6, color=C_ACTUAL)
    ax.plot(d, p, label=label, alpha=0.9, linewidth=1.1, color=color)
    if sig is not None:
        ax.fill_between(d, p - sig, p + sig, alpha=0.12, color=color)
    ax.set_ylabel('Log Return')
    ax.set_title(f'{label}   (RMSE = {rmse:.5f},  Dir. Acc. = {da:.1%},  Pred. Std = {p.std():.5f})')
    ax.legend(loc='upper right')
    ax.grid(True)
format_date_axis(axes[2])
axes[2].set_xlabel('Date')
plt.tight_layout()
plt.savefig(FINAL_DIR / '06a_nll_individual_predictions.png')
plt.close()
print('  06a')

# ══════════════════════════════════════════════════════════════
# 06b: NLL all models combined
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(test_dates, acts, label='Actual', alpha=0.45, linewidth=0.6, color=C_ACTUAL)
ax.plot(test_dates, preds_nll, label='LSTM (NLL)', alpha=0.7, linewidth=0.9, color=C_NLL)
ax.plot(test_dates, xgb_preds, label='XGBoost', alpha=0.7, linewidth=0.9, color=C_XGB)
ax.plot(ar_test_dates, ar_preds, label='AR(10)', alpha=0.7, linewidth=0.9, color=C_AR)
ax.set_xlabel('Date'); ax.set_ylabel('Log Return')
ax.set_title('All Models vs Actual (Test Set) — NLL Framework')
ax.legend(loc='upper right'); ax.grid(True)
format_date_axis(ax)
plt.tight_layout()
plt.savefig(FINAL_DIR / '06b_nll_all_combined.png')
plt.close()
print('  06b')

# ══════════════════════════════════════════════════════════════
# 06c: NLL scatter predicted vs actual (3 panels)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
scatter = [
    (acts, preds_nll, 'LSTM (NLL)', C_NLL),
    (acts, xgb_preds, 'XGBoost', C_XGB),
    (ar_y_te, ar_preds, 'AR(10)', C_AR),
]
for ax, (a, p, label, color) in zip(axes, scatter):
    ax.scatter(a, p, alpha=0.35, s=12, color=color, edgecolors='none')
    lims = [min(a.min(), p.min()) * 1.1, max(a.max(), p.max()) * 1.1]
    ax.plot(lims, lims, color='#999999', linewidth=0.8, linestyle='--', label='y = x')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(label); ax.legend(fontsize=8); ax.grid(True)
    ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(FINAL_DIR / '06c_nll_scatter.png')
plt.close()
print('  06c')

# ══════════════════════════════════════════════════════════════
# 06d: NLL residual analysis (3x3)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
residuals = [
    (acts - preds_nll, test_dates, 'LSTM (NLL)', C_NLL),
    (acts - xgb_preds, test_dates, 'XGBoost', C_XGB),
    (ar_y_te - ar_preds, ar_test_dates, 'AR(10)', C_AR),
]
for row, (resid, d, label, color) in enumerate(residuals):
    axes[row, 0].plot(d, resid, linewidth=0.5, color=color, alpha=0.8)
    axes[row, 0].axhline(0, color='#333333', linewidth=0.5)
    axes[row, 0].set_title(f'{label} Residuals', fontsize=10)
    axes[row, 0].set_ylabel('Residual'); axes[row, 0].grid(True)
    if row == 2: format_date_axis(axes[row, 0])

    axes[row, 1].hist(resid, bins=50, color=color, alpha=0.7,
                      edgecolor='white', linewidth=0.3, density=True)
    axes[row, 1].set_title(f'Distribution (std = {resid.std():.4f})', fontsize=10)
    axes[row, 1].set_xlabel('Residual'); axes[row, 1].grid(True)

    m = len(resid); rc = resid - resid.mean()
    acf_f = np.correlate(rc, rc, mode='full')
    acf_v = acf_f[m-1:m+21] / acf_f[m-1]
    ci_r = 1.96 / np.sqrt(m)
    axes[row, 2].bar(range(len(acf_v)), acf_v, color=color, alpha=0.7,
                     width=0.6, edgecolor='white', linewidth=0.3)
    axes[row, 2].axhline(ci_r, color='#999999', linestyle='--', linewidth=0.8)
    axes[row, 2].axhline(-ci_r, color='#999999', linestyle='--', linewidth=0.8)
    axes[row, 2].set_title('Residual ACF', fontsize=10)
    axes[row, 2].set_xlabel('Lag'); axes[row, 2].grid(True)
plt.tight_layout()
plt.savefig(FINAL_DIR / '06d_nll_residual_analysis.png')
plt.close()
print('  06d')

# ══════════════════════════════════════════════════════════════
# 06e: NLL learned uncertainty
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(test_dates, sigma_nll, linewidth=0.9, color=C_NLL, label='Learned \u03c3', alpha=0.85)
ax.axhline(acts.std(), color='#999999', linestyle='--', linewidth=0.8,
           label=f'Actual std ({acts.std():.4f})')
ax.set_ylabel('Learned \u03c3'); ax.set_xlabel('Date')
ax.set_title('LSTM (NLL) — Per-Observation Learned Uncertainty')
ax.legend(); ax.grid(True)
format_date_axis(ax)
plt.tight_layout()
plt.savefig(FINAL_DIR / '06e_nll_learned_uncertainty.png')
plt.close()
print('  06e')

# ══════════════════════════════════════════════════════════════
# 06f: NLL training curves
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, len(tl_nll)+1), tl_nll, label='Train (NLL)', linewidth=1.2, color=C_NLL)
ax.plot(range(1, len(vl_nll)+1), vl_nll, label='Val (NLL)', linewidth=1.2, color=C_ACTUAL, linestyle='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('LSTM (Pure NLL) — Training Curves')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(FINAL_DIR / '06f_nll_training_curves.png')
plt.close()
print('  06f')


print('\n--- Generating custom loss plots (07a-07f) ---')

# ══════════════════════════════════════════════════════════════
# 07a: Ablation — Pure NLL → NLL+Dir → NLL+Dir+Var
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
ablation = [
    (preds_nll, sigma_nll, 'Pure NLL', C_NLL),
    (preds_dir, sigma_dir, 'NLL + Directional Penalty', C_DIR),
    (preds_combo, sigma_combo, 'NLL + Directional + Variance Penalty', C_COMBO),
]
for i, (preds, sigma, label, color) in enumerate(ablation):
    ax = axes[i]
    rmse = np.sqrt(mean_squared_error(acts, preds))
    da = np.mean(np.sign(preds) == np.sign(acts))
    ax.plot(test_dates, acts, label='Actual', alpha=0.5, linewidth=0.6, color=C_ACTUAL)
    ax.plot(test_dates, preds, label=label, alpha=0.9, linewidth=1.1, color=color)
    ax.fill_between(test_dates, preds - sigma, preds + sigma, alpha=0.12, color=color)
    ax.set_ylabel('Log Return')
    ax.set_title(f'{label}   (RMSE = {rmse:.5f},  Dir. Acc. = {da:.1%},  Pred. Std = {preds.std():.5f})')
    ax.legend(loc='upper right'); ax.grid(True)
format_date_axis(axes[2])
axes[2].set_xlabel('Date')
plt.tight_layout()
plt.savefig(FINAL_DIR / '07a_ablation_nll_dir_var.png')
plt.close()
print('  07a')

# ══════════════════════════════════════════════════════════════
# 07b: NLL+Dir+Var with confidence band (hero plot)
# ══════════════════════════════════════════════════════════════
rmse_c = np.sqrt(mean_squared_error(acts, preds_combo))
da_c = np.mean(np.sign(preds_combo) == np.sign(acts))
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(test_dates, acts, label='Actual', alpha=0.5, linewidth=0.6, color=C_ACTUAL)
ax.plot(test_dates, preds_combo, label='Prediction (\u03bc)', alpha=0.9, linewidth=1.1, color=C_COMBO)
ax.fill_between(test_dates, preds_combo - sigma_combo, preds_combo + sigma_combo,
                alpha=0.18, color=C_COMBO, label='\u00b11\u03c3 confidence')
ax.set_xlabel('Date'); ax.set_ylabel('Log Return')
ax.set_title(f'LSTM (NLL + Dir + Var) with Learned Uncertainty   (RMSE = {rmse_c:.5f},  Dir. Acc. = {da_c:.1%})')
ax.legend(); ax.grid(True)
format_date_axis(ax)
plt.tight_layout()
plt.savefig(FINAL_DIR / '07b_custom_loss_with_uncertainty.png')
plt.close()
print('  07b')

# ══════════════════════════════════════════════════════════════
# 07c: Before vs after — NLL vs NLL+Dir+Var side by side
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(test_dates, acts, label='Actual', alpha=0.5, linewidth=0.6, color=C_ACTUAL)
axes[0].plot(test_dates, preds_nll, label='LSTM (Pure NLL)', alpha=0.9, linewidth=1.1, color=C_NLL)
rmse_n = np.sqrt(mean_squared_error(acts, preds_nll))
da_n = np.mean(np.sign(preds_nll) == np.sign(acts))
axes[0].set_title(f'Before: Pure NLL   (RMSE = {rmse_n:.5f},  Dir. Acc. = {da_n:.1%},  Pred. Std = {preds_nll.std():.5f})')
axes[0].set_ylabel('Log Return'); axes[0].legend(loc='upper right'); axes[0].grid(True)

axes[1].plot(test_dates, acts, label='Actual', alpha=0.5, linewidth=0.6, color=C_ACTUAL)
axes[1].plot(test_dates, preds_combo, label='LSTM (NLL+Dir+Var)', alpha=0.9, linewidth=1.1, color=C_COMBO)
axes[1].set_title(f'After: NLL + Dir + Var   (RMSE = {rmse_c:.5f},  Dir. Acc. = {da_c:.1%},  Pred. Std = {preds_combo.std():.5f})')
axes[1].set_ylabel('Log Return'); axes[1].set_xlabel('Date')
axes[1].legend(loc='upper right'); axes[1].grid(True)
format_date_axis(axes[1])
plt.tight_layout()
plt.savefig(FINAL_DIR / '07c_before_vs_after.png')
plt.close()
print('  07c')

# ══════════════════════════════════════════════════════════════
# 07d: Custom loss scatter (NLL+Dir+Var vs actual)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (a, p, label, color) in zip(axes, [
    (acts, preds_nll, 'LSTM (Pure NLL)', C_NLL),
    (acts, preds_combo, 'LSTM (NLL+Dir+Var)', C_COMBO),
]):
    ax.scatter(a, p, alpha=0.35, s=12, color=color, edgecolors='none')
    lims = [min(a.min(), p.min()) * 1.1, max(a.max(), p.max()) * 1.1]
    ax.plot(lims, lims, color='#999999', linewidth=0.8, linestyle='--', label='y = x')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(label); ax.legend(fontsize=8); ax.grid(True)
    ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(FINAL_DIR / '07d_custom_loss_scatter.png')
plt.close()
print('  07d')

# ══════════════════════════════════════════════════════════════
# 07e: Custom loss training curves
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, len(tl_combo)+1), tl_combo, label='Train (composite)', linewidth=1.2, color=C_COMBO)
ax.plot(range(1, len(vl_combo)+1), vl_combo, label='Val (Pure NLL)', linewidth=1.2, color=C_ACTUAL, linestyle='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('LSTM (NLL+Dir+Var) — Training Curves')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(FINAL_DIR / '07e_custom_loss_training_curves.png')
plt.close()
print('  07e')

# ══════════════════════════════════════════════════════════════
# 07f: Learned uncertainty comparison (NLL vs NLL+Dir+Var)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(test_dates, sigma_nll, linewidth=0.9, color=C_NLL, label='Pure NLL', alpha=0.85)
ax.plot(test_dates, sigma_combo, linewidth=0.9, color=C_COMBO, label='NLL+Dir+Var', alpha=0.85)
ax.axhline(acts.std(), color='#999999', linestyle='--', linewidth=0.8,
           label=f'Actual std ({acts.std():.4f})')
ax.set_ylabel('Learned \u03c3'); ax.set_xlabel('Date')
ax.set_title('Learned Uncertainty Comparison')
ax.legend(fontsize=8); ax.grid(True)
format_date_axis(ax)
plt.tight_layout()
plt.savefig(FINAL_DIR / '07f_uncertainty_comparison.png')
plt.close()
print('  07f')

print('\nDone. All plots in results/final/')
for f in sorted(FINAL_DIR.glob('*.png')):
    print(f'  {f.name}')
