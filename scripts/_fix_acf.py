"""Regenerate plots 02 and 13 with ACF lag 0 included."""
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
from sklearn.metrics import mean_squared_error
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
C_NLL = '#c0392b'
C_COMBO = '#8e44ad'
C_XGB = '#27ae60'
FINAL_DIR = Path('results/final')
SEED = 42

df = pd.read_parquet('data/processed/sanity_data_pca.parquet')
df['Date'] = pd.to_datetime(df['Date'])
TARGET = 'Target'
NON_FEATURE_COLS = ['Date', 'Target', 'WTI_Close', 'Brent_Close']
FEATURE_COLS = [c for c in df.columns if c not in NON_FEATURE_COLS]
df = df.dropna(subset=[TARGET] + FEATURE_COLS).reset_index(drop=True)

# ── Plot 02 ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(df[TARGET], bins=100, color=C_ACTUAL, alpha=0.75,
             edgecolor='white', linewidth=0.3, density=True)
axes[0].set_xlabel('Log Return')
axes[0].set_ylabel('Density')
axes[0].set_title(
    f'Return Distribution (skew = {df[TARGET].skew():.2f}, '
    f'kurtosis = {df[TARGET].kurtosis():.1f})')
axes[0].grid(True)

returns = df[TARGET].values
n = len(returns)
r = returns - returns.mean()
acf_full = np.correlate(r, r, mode='full')
acf_vals = acf_full[n-1:n+31] / acf_full[n-1]
ci = 1.96 / np.sqrt(n)
axes[1].bar(range(len(acf_vals)), acf_vals, color=C_ACTUAL, alpha=0.7,
            width=0.7, edgecolor='white', linewidth=0.3)
axes[1].axhline(ci, color=C_NLL, linestyle='--', linewidth=0.8, label='95% CI')
axes[1].axhline(-ci, color=C_NLL, linestyle='--', linewidth=0.8)
axes[1].set_xlabel('Lag (days)')
axes[1].set_ylabel('Autocorrelation')
axes[1].set_title('Return Autocorrelation Function')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig(FINAL_DIR / '02_return_distribution_and_acf.png', dpi=150, bbox_inches='tight')
plt.close()
print('02 done')

# ── Setup for plot 13 ──
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

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def train_pred(sl, h, l, d, lr, loss_fn):
    Xtr, ytr = create_sequences(train_X, train_y, sl)
    Xvl, yvl = create_sequences_with_bridge(train_X, train_y, val_X, val_y, sl)
    Xte, yte = create_sequences_with_bridge(val_X, val_y, test_X, test_y, sl)
    trl = DataLoader(OilDataset(Xtr, ytr), batch_size=32, shuffle=False)
    vll = DataLoader(OilDataset(Xvl, yvl), batch_size=32, shuffle=False)
    tel = DataLoader(OilDataset(Xte, yte), batch_size=32, shuffle=False)
    torch.manual_seed(SEED)
    model = LSTMModelNLL(n_features, h, l, d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bv, bs, w = float('inf'), None, 0
    for ep in range(60):
        model.train()
        for Xb, yb in trl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            mu, lv = model(Xb)
            loss = loss_fn(mu, lv, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vl = 0
        with torch.no_grad():
            for Xb, yb in vll:
                Xb, yb = Xb.to(device), yb.to(device)
                mu, lv = model(Xb)
                vl += pure_nll(mu, lv, yb).item()
        av = vl / len(vll)
        if av < bv:
            bv = av
            bs = {k: v.clone() for k, v in model.state_dict().items()}
            w = 0
        else:
            w += 1
            if w >= 15:
                break
    model.load_state_dict(bs)
    model.eval()
    ms, acts = [], []
    with torch.no_grad():
        for Xb, yb in tel:
            Xb = Xb.to(device)
            mu, _ = model(Xb)
            ms.append(mu.cpu().numpy())
            acts.append(yb.numpy())
    ms = np.concatenate(ms).flatten()
    acts = np.concatenate(acts).flatten()
    preds = scaler_y.inverse_transform(ms.reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(acts.reshape(-1, 1)).flatten()
    return preds, actuals

print('Training models for residual plot...')
pn, acts = train_pred(3, 128, 3, 0.0, 1e-3, pure_nll)
pc, _ = train_pred(5, 128, 2, 0.3, 5e-4,
                   lambda mu, lv, t: nll_dir_var(mu, lv, t))

xm = XGBRegressor(n_estimators=50, max_depth=7, learning_rate=0.01,
                   subsample=0.9, colsample_bytree=0.7, reg_alpha=0,
                   reg_lambda=2, objective='reg:squarederror',
                   random_state=SEED, verbosity=0, early_stopping_rounds=20)
xm.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
xp = scaler_y.inverse_transform(
    xm.predict(test_X).reshape(-1, 1)).flatten()[-len(acts):]

test_dates = test_df['Date'].values[-len(acts):]

# ── Plot 13 ──
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
residuals = [
    (acts - pn, test_dates, 'LSTM (Pure NLL)', C_NLL),
    (acts - pc, test_dates, 'LSTM (NLL+Dir+Var)', C_COMBO),
    (acts - xp, test_dates, 'XGBoost', C_XGB),
]
for row, (resid, d, label, color) in enumerate(residuals):
    axes[row, 0].plot(d, resid, linewidth=0.5, color=color, alpha=0.8)
    axes[row, 0].axhline(0, color='#333333', linewidth=0.5)
    axes[row, 0].set_title(f'{label} Residuals', fontsize=10)
    axes[row, 0].set_ylabel('Residual')
    axes[row, 0].grid(True)
    if row == 2:
        axes[row, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[row, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(axes[row, 0].xaxis.get_majorticklabels(), rotation=45,
                 ha='right', fontsize=8)

    axes[row, 1].hist(resid, bins=50, color=color, alpha=0.7,
                      edgecolor='white', linewidth=0.3, density=True)
    axes[row, 1].set_title(f'Distribution (std = {resid.std():.4f})', fontsize=10)
    axes[row, 1].set_xlabel('Residual')
    axes[row, 1].grid(True)

    m = len(resid)
    rc = resid - resid.mean()
    acf_f = np.correlate(rc, rc, mode='full')
    acf_v = acf_f[m-1:m+21] / acf_f[m-1]
    axes[row, 2].bar(range(len(acf_v)), acf_v, color=color, alpha=0.7,
                     width=0.6, edgecolor='white', linewidth=0.3)
    ci_r = 1.96 / np.sqrt(m)
    axes[row, 2].axhline(ci_r, color='#999999', linestyle='--', linewidth=0.8)
    axes[row, 2].axhline(-ci_r, color='#999999', linestyle='--', linewidth=0.8)
    axes[row, 2].set_title('Residual ACF', fontsize=10)
    axes[row, 2].set_xlabel('Lag')
    axes[row, 2].grid(True)

plt.tight_layout()
plt.savefig(FINAL_DIR / '13_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('13 done')
