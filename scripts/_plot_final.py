"""Plot final model predictions vs actuals: individual + combined."""
import pandas as pd
import numpy as np
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
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
plt.style.use('seaborn-v0_8')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROJECT_DIR = Path('.')
RESULTS_DIR = PROJECT_DIR / 'results' / 'nll_grid'

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


def train_and_predict_lstm(sl, hidden, layers, dropout, lr, loss_fn, label):
    X_tr, y_tr = create_sequences(train_X, train_y, sl)
    X_vl, y_vl = create_sequences_with_bridge(train_X, train_y, val_X, val_y, sl)
    X_te, y_te = create_sequences_with_bridge(val_X, val_y, test_X, test_y, sl)
    tr_loader = DataLoader(OilDataset(X_tr, y_tr), batch_size=32, shuffle=False)
    vl_loader = DataLoader(OilDataset(X_vl, y_vl), batch_size=32, shuffle=False)
    te_loader = DataLoader(OilDataset(X_te, y_te), batch_size=32, shuffle=False)

    torch.manual_seed(SEED)
    model = LSTMModelNLL(n_features, hidden, layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, wait = float('inf'), None, 0
    for epoch in range(60):
        model.train()
        for X_b, y_b in tr_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            mu, lv = model(X_b)
            loss = loss_fn(mu, lv, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        vl = 0
        with torch.no_grad():
            for X_b, y_b in vl_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                mu, lv = model(X_b)
                vl += pure_nll(mu, lv, y_b).item()
        avg_vl = vl / len(vl_loader)
        if avg_vl < best_val:
            best_val = avg_vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 15:
                break
    model.load_state_dict(best_state)
    model.eval()
    mus, logvars, actuals = [], [], []
    with torch.no_grad():
        for X_b, y_b in te_loader:
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
    dir_acc = np.mean(np.sign(preds) == np.sign(acts))
    rmse = np.sqrt(mean_squared_error(acts, preds))
    print(f'{label}: RMSE={rmse:.6f} Dir={dir_acc:.4f} PredStd={preds.std():.6f}')
    return preds, acts, sigma


# ── Generate predictions ──
print('Training models...')

# LSTM Pure NLL (best: sl=3, h=128, L=3, d=0.0, lr=1e-3)
preds_nll, acts, sigma_nll = train_and_predict_lstm(
    3, 128, 3, 0.0, 1e-3, pure_nll, 'Pure NLL')

# LSTM NLL+Dir+Var (best: sl=5, h=128, L=2, d=0.3, lr=5e-4)
preds_combo, _, sigma_combo = train_and_predict_lstm(
    5, 128, 2, 0.3, 5e-4,
    lambda mu, lv, t: nll_dir_var(mu, lv, t, beta=1.0, gamma=0.05),
    'NLL+Dir+Var')

# XGBoost (best: n_est=50, depth=7, lr=0.01, sub=0.9, col=0.7, alpha=0, lambda=2)
xgb_model = XGBRegressor(n_estimators=50, max_depth=7, learning_rate=0.01,
                          subsample=0.9, colsample_bytree=0.7, reg_alpha=0.0,
                          reg_lambda=2.0, objective='reg:squarederror',
                          random_state=SEED, verbosity=0, early_stopping_rounds=20)
xgb_model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
xgb_preds_all = scaler_y.inverse_transform(xgb_model.predict(test_X).reshape(-1,1)).flatten()
lstm_n = len(acts)
xgb_preds = xgb_preds_all[-lstm_n:]
xgb_dir = np.mean(np.sign(xgb_preds) == np.sign(acts))
xgb_rmse = np.sqrt(mean_squared_error(acts, xgb_preds))
print(f'XGBoost: RMSE={xgb_rmse:.6f} Dir={xgb_dir:.4f} PredStd={xgb_preds.std():.6f}')

# AR(10)
ar_order = 10
target_vals = df[TARGET].values
ar_X_all = np.column_stack([np.roll(target_vals, i) for i in range(1, ar_order + 1)])
ar_y_all = target_vals.copy()
valid_idx = np.arange(ar_order, len(target_vals))
ar_X_all, ar_y_all = ar_X_all[valid_idx], ar_y_all[valid_idx]
ar_dates_all = df['Date'].values[valid_idx]
train_mask = ar_dates_all <= np.datetime64('2022-12-31')
test_mask = ar_dates_all > np.datetime64('2024-06-30')
ar_X_tr, ar_y_tr = ar_X_all[train_mask], ar_y_all[train_mask]
ar_X_te, ar_y_te = ar_X_all[test_mask], ar_y_all[test_mask]
ar_test_dates = ar_dates_all[test_mask]
beta_ar = np.linalg.lstsq(np.column_stack([np.ones(len(ar_X_tr)), ar_X_tr]), ar_y_tr, rcond=None)[0]
ar_preds = np.column_stack([np.ones(len(ar_X_te)), ar_X_te]) @ beta_ar
print(f'AR(10): RMSE={np.sqrt(mean_squared_error(ar_y_te, ar_preds)):.6f} '
      f'Dir={np.mean(np.sign(ar_preds)==np.sign(ar_y_te)):.4f}')

# Dates for LSTM test window
test_dates = test_df['Date'].values[-lstm_n:]


# ── PLOT 1: Individual panels (4 rows) ──
print('\nGenerating plots...')
fig, axes = plt.subplots(4, 1, figsize=(14, 18))

models = [
    (preds_nll, sigma_nll, 'LSTM — Pure NLL', '#c55a11',
     f'RMSE={np.sqrt(mean_squared_error(acts, preds_nll)):.5f}, '
     f'Dir={np.mean(np.sign(preds_nll)==np.sign(acts)):.3f}, '
     f'PredStd={preds_nll.std():.5f}'),
    (preds_combo, sigma_combo, 'LSTM — NLL+Dir+Var', '#8e44ad',
     f'RMSE={np.sqrt(mean_squared_error(acts, preds_combo)):.5f}, '
     f'Dir={np.mean(np.sign(preds_combo)==np.sign(acts)):.3f}, '
     f'PredStd={preds_combo.std():.5f}'),
    (xgb_preds, None, 'XGBoost — MSE', '#2ca02c',
     f'RMSE={xgb_rmse:.5f}, Dir={xgb_dir:.3f}, PredStd={xgb_preds.std():.5f}'),
    (ar_preds, None, 'AR(10) — OLS', '#375623',
     f'RMSE={np.sqrt(mean_squared_error(ar_y_te, ar_preds)):.5f}, '
     f'Dir={np.mean(np.sign(ar_preds)==np.sign(ar_y_te)):.3f}, '
     f'PredStd={ar_preds.std():.5f}'),
]

for i, (preds, sigma, title, color, stats) in enumerate(models):
    ax = axes[i]
    if i < 3:  # LSTM and XGBoost use test_dates
        ax.plot(test_dates, acts, label='Actual', alpha=0.6, linewidth=0.8, color='#1f4e79')
        ax.plot(test_dates, preds, label=title, alpha=0.9, linewidth=1.2, color=color)
        if sigma is not None:
            ax.fill_between(test_dates, preds - sigma, preds + sigma,
                            alpha=0.15, color=color, label='\u00b11\u03c3')
    else:  # AR uses its own dates
        ax.plot(ar_test_dates, ar_y_te, label='Actual', alpha=0.6, linewidth=0.8, color='#1f4e79')
        ax.plot(ar_test_dates, ar_preds, label=title, alpha=0.9, linewidth=1.2, color=color)
    ax.set_ylabel('Log Return')
    ax.set_title(f'{title}  ({stats})')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

axes[3].set_xlabel('Date')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'predictions_individual.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: predictions_individual.png')


# ── PLOT 2: All combined ──
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_dates, acts, label='Actual', alpha=0.6, linewidth=0.8, color='#1f4e79')
ax.plot(test_dates, preds_nll, label='LSTM (Pure NLL)', alpha=0.8, linewidth=1, color='#c55a11')
ax.plot(test_dates, preds_combo, label='LSTM (NLL+Dir+Var)', alpha=0.8, linewidth=1, color='#8e44ad')
ax.plot(test_dates, xgb_preds, label='XGBoost (MSE)', alpha=0.8, linewidth=1, color='#2ca02c')
ax.plot(ar_test_dates, ar_preds, label='AR(10)', alpha=0.8, linewidth=1, color='#375623')
ax.set_xlabel('Date')
ax.set_ylabel('Log Return')
ax.set_title('All Models — Predictions vs Actual (Test Set)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'predictions_combined.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: predictions_combined.png')


# ── PLOT 3: NLL+Dir+Var with confidence band (the interesting one) ──
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_dates, acts, label='Actual', alpha=0.6, linewidth=0.8, color='#1f4e79')
ax.plot(test_dates, preds_combo, label='Prediction (\u03bc)', alpha=0.9, linewidth=1.2, color='#8e44ad')
ax.fill_between(test_dates, preds_combo - sigma_combo, preds_combo + sigma_combo,
                alpha=0.2, color='#8e44ad', label='\u00b11\u03c3 confidence')
ax.set_xlabel('Date')
ax.set_ylabel('Log Return')
ax.set_title('LSTM (NLL+Dir+Var) — Predictions with Learned Uncertainty')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'predictions_nll_dir_var_uncertainty.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: predictions_nll_dir_var_uncertainty.png')

print('\nDone.')
