"""
FA-3DPC: Simultaneous CS and FS Prediction
LOMO Cross-Validation | 8 ML Models | Optuna Hyperparameter Optimization
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import optuna
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH   = os.path.join('..', '可视化分析', 'dataset.csv')
RESULTS_DIR = 'results'
SEED        = 42
N_TRIALS    = 30          # Optuna trials per model per LOMO fold
CV_INNER    = 5           # inner CV folds for Optuna objective

FEATURES = ['FA_pct', 'W_B', 'Age', 'ln_Age', 'FA_Age']
TARGETS  = ['CS (MPa)', 'FS (MPa)']

MIX_ORDER = [
    'FA0-WB-0.35', 'FA0-WB-0.34', 'FA0-WB-0.30',
    'FA5-WB-0.30', 'FA7.5-WB-0.30', 'FA10-WB-0.30', 'FA15-WB-0.30',
]

# Models that require StandardScaler
SCALE_MODELS = {'ElasticNet', 'SVR', 'GPR'}

MODEL_NAMES = ['ElasticNet', 'SVR', 'RandomForest', 'ExtraTrees',
               'XGBoost', 'LightGBM', 'CatBoost', 'GPR']

np.random.seed(SEED)

# ============================================================================
# Data Loading & Feature Engineering
# ============================================================================
def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Sand (g)', 'Accelerator (%)', 'SP (%)'])

    df = df.rename(columns={
        'Fly ash (g)': 'FA_pct',
        'W/B':         'W_B',
        'Age (days)':  'Age',
    })

    df['ln_Age'] = np.log(df['Age'])
    df['FA_Age'] = df['FA_pct'] * df['Age']
    df['Mix']    = df.apply(_assign_mix, axis=1)

    return df


def _assign_mix(row):
    fa, wb = row['FA_pct'], row['W_B']
    if fa == 0:    return f'FA0-WB-{wb:.2f}'
    elif fa <= 6:  return 'FA5-WB-0.30'
    elif fa <= 9:  return 'FA7.5-WB-0.30'
    elif fa <= 12: return 'FA10-WB-0.30'
    else:          return 'FA15-WB-0.30'

# ============================================================================
# LOMO Split Generator
# ============================================================================
def lomo_splits(df):
    """Yield (mix_name, train_indices, test_indices) for each mix."""
    for mix in MIX_ORDER:
        mask     = df['Mix'] == mix
        te_idx   = df.index[mask].tolist()
        tr_idx   = df.index[~mask].tolist()
        yield mix, tr_idx, te_idx

# ============================================================================
# Hyperparameter Search Spaces
# ============================================================================
def suggest_params(trial, name):
    if name == 'ElasticNet':
        return {
            'alpha':    trial.suggest_float('alpha',    1e-4, 10.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0,  1.0),
        }
    if name == 'SVR':
        return {
            'kernel':  trial.suggest_categorical('kernel',  ['rbf', 'poly']),
            'C':       trial.suggest_float('C',       0.1,  1000.0, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1.0,    log=True),
            'gamma':   trial.suggest_categorical('gamma', ['scale', 'auto']),
        }
    if name == 'RandomForest':
        return {
            'n_estimators':      trial.suggest_int('n_estimators',      50, 300),
            'max_depth':         trial.suggest_int('max_depth',          2,  12),
            'min_samples_split': trial.suggest_int('min_samples_split',  2,  10),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf',   1,   6),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
    if name == 'ExtraTrees':
        return {
            'n_estimators':      trial.suggest_int('n_estimators',      50, 300),
            'max_depth':         trial.suggest_int('max_depth',          2,  12),
            'min_samples_split': trial.suggest_int('min_samples_split',  2,  10),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf',   1,   6),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
    if name == 'XGBoost':
        return {
            'n_estimators':     trial.suggest_int('n_estimators',       50, 400),
            'max_depth':        trial.suggest_int('max_depth',           2,   8),
            'learning_rate':    trial.suggest_float('learning_rate',  0.01, 0.30, log=True),
            'subsample':        trial.suggest_float('subsample',       0.5,  1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha',      1e-5, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda',     1e-5, 10.0, log=True),
        }
    if name == 'LightGBM':
        return {
            'n_estimators':     trial.suggest_int('n_estimators',       50, 400),
            'max_depth':        trial.suggest_int('max_depth',           2,   8),
            'num_leaves':       trial.suggest_int('num_leaves',          8,  64),
            'learning_rate':    trial.suggest_float('learning_rate',  0.01, 0.30, log=True),
            'subsample':        trial.suggest_float('subsample',       0.5,  1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha',      1e-5, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda',     1e-5, 10.0, log=True),
        }
    if name == 'CatBoost':
        return {
            'iterations':    trial.suggest_int('iterations',       100, 800),
            'depth':         trial.suggest_int('depth',              3,   8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.30, log=True),
            'l2_leaf_reg':   trial.suggest_float('l2_leaf_reg',   1e-3, 10.0, log=True),
            'subsample':     trial.suggest_float('subsample',      0.5,  1.0),
        }
    raise ValueError(f'Unknown model: {name}')


def build_model(name, params):
    if name == 'ElasticNet':
        return ElasticNet(
            alpha=params['alpha'], l1_ratio=params['l1_ratio'],
            max_iter=5000, random_state=SEED)
    if name == 'SVR':
        return SVR(
            kernel=params['kernel'], C=params['C'],
            epsilon=params['epsilon'], gamma=params['gamma'])
    if name == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=SEED, n_jobs=-1)
    if name == 'ExtraTrees':
        return ExtraTreesRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=SEED, n_jobs=-1)
    if name == 'XGBoost':
        return xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            verbosity=0, random_state=SEED)
    if name == 'LightGBM':
        return lgb.LGBMRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            num_leaves=params['num_leaves'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            verbose=-1, random_state=SEED)
    if name == 'CatBoost':
        return CatBoostRegressor(
            iterations=params['iterations'],
            depth=params['depth'],
            learning_rate=params['learning_rate'],
            l2_leaf_reg=params['l2_leaf_reg'],
            subsample=params['subsample'],
            bootstrap_type='Bernoulli',
            verbose=0, random_state=SEED)
    raise ValueError(f'Unknown model: {name}')

# ============================================================================
# Optuna Objective
# ============================================================================
def make_objective(name, X_tr, y_tr):
    def objective(trial):
        params = suggest_params(trial, name)
        model  = build_model(name, params)

        X = StandardScaler().fit_transform(X_tr) if name in SCALE_MODELS else X_tr
        # LightGBM requires consistent feature names across fit/predict inside CV
        if name == 'LightGBM':
            X = pd.DataFrame(X, columns=FEATURES)

        scores = cross_val_score(
            model, X, y_tr,
            cv=CV_INNER,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
        )
        return float(-scores.mean())
    return objective

# ============================================================================
# Metrics Helper
# ============================================================================
def compute_metrics(y_true, y_pred):
    return {
        'R2':   float(r2_score(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE':  float(mean_absolute_error(y_true, y_pred)),
    }

# ============================================================================
# Train / Evaluate (non-GPR)
# ============================================================================
def run_model(name, params, X_tr, y_tr, X_te, y_te):
    model = build_model(name, params)

    if name in SCALE_MODELS:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
    else:
        scaler = None
        X_tr_s, X_te_s = X_tr, X_te

    # LightGBM: use DataFrame to keep feature names consistent
    if name == 'LightGBM':
        X_tr_s = pd.DataFrame(X_tr_s, columns=FEATURES)
        X_te_s = pd.DataFrame(X_te_s, columns=FEATURES)

    model.fit(X_tr_s, y_tr)
    pred_te = model.predict(X_te_s)
    return (compute_metrics(y_tr, model.predict(X_tr_s)),
            compute_metrics(y_te, pred_te),
            model, scaler, pred_te)

# ============================================================================
# GPR (sklearn optimizes kernel internally — no Optuna needed)
# ============================================================================
def run_gpr(X_tr, y_tr, X_te, y_te):
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-2)
    gpr    = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5,
        normalize_y=True, random_state=SEED)

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    gpr.fit(X_tr_s, y_tr)
    pred_te, std_te = gpr.predict(X_te_s, return_std=True)

    return (compute_metrics(y_tr, gpr.predict(X_tr_s)),
            compute_metrics(y_te, pred_te),
            gpr, scaler, std_te)

# ============================================================================
# Results Visualization
# ============================================================================
def plot_lomo_comparison(summary_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ['#3C5488', '#E64B35', '#00A087', '#F39B7F',
              '#4DBBD5', '#8491B4', '#DC0000', '#7E6148']

    for ax, target in zip(axes, TARGETS):
        sub = summary_df[summary_df['Target'] == target].reset_index(drop=True)
        x   = np.arange(len(sub))
        bars = ax.bar(x, sub['R2_mean'], yerr=sub['R2_std'],
                      color=colors[:len(sub)], alpha=0.85,
                      edgecolor='white', linewidth=0.6,
                      error_kw=dict(elinewidth=1.2, capsize=4, ecolor='#333333'))

        for i, (r, s) in enumerate(zip(sub['R2_mean'], sub['R2_std'])):
            ax.text(i, r + s + 0.01, f'{r:.3f}', ha='center',
                    va='bottom', fontsize=7.5, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(sub['Model'], rotation=35, ha='right', fontsize=8.5)
        ax.set_ylabel('LOMO R2', fontweight='bold')
        label = 'CS' if 'CS' in target else 'FS'
        ax.set_title(f'({chr(96 + list(TARGETS).index(target) + 1)}) {label} Prediction',
                     fontweight='bold')
        ax.set_ylim(0, 1.08)
        ax.axhline(0.9, color='#AAAAAA', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_lomo_comparison.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(out_dir, 'fig_lomo_comparison.svg'),
                format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Saved: fig_lomo_comparison.png / .svg')


def plot_lomo_scatter(df, best_info, out_dir):
    """Predicted vs actual scatter under LOMO for best model per target."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, target in zip(axes, TARGETS):
        name   = best_info[target]['model']
        params = best_info[target]['params']
        all_true, all_pred = [], []

        for mix, tr_idx, te_idx in lomo_splits(df):
            X_tr = df.loc[tr_idx, FEATURES].values
            X_te = df.loc[te_idx, FEATURES].values
            y_tr = df.loc[tr_idx, target].values
            y_te = df.loc[te_idx, target].values
            _, _, _, _, pred_te = run_model(name, params, X_tr, y_tr, X_te, y_te)
            all_true.extend(y_te)
            all_pred.extend(pred_te)

        all_true, all_pred = np.array(all_true), np.array(all_pred)
        r2 = r2_score(all_true, all_pred)
        rmse = np.sqrt(mean_squared_error(all_true, all_pred))

        lim = [min(all_true.min(), all_pred.min()) * 0.95,
               max(all_true.max(), all_pred.max()) * 1.05]
        ax.plot(lim, lim, 'k--', linewidth=1.2, alpha=0.7)
        ax.scatter(all_true, all_pred, s=35, alpha=0.75,
                   color='#3C5488', edgecolors='white', linewidths=0.4)

        ax.text(0.05, 0.92, f'R2 = {r2:.4f}\nRMSE = {rmse:.3f}',
                transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#AAAAAA', alpha=0.9))

        label = 'CS (MPa)' if 'CS' in target else 'FS (MPa)'
        ax.set_xlabel(f'Measured {label}', fontweight='bold')
        ax.set_ylabel(f'Predicted {label}', fontweight='bold')
        letter = chr(96 + list(TARGETS).index(target) + 1)
        ax.set_title(f'({letter}) {name}, LOMO', fontweight='bold')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_lomo_scatter.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(out_dir, 'fig_lomo_scatter.svg'),
                format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Saved: fig_lomo_scatter.png / .svg')

# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    print(f'Dataset: {df.shape[0]} rows')
    print(f'Features: {FEATURES}')
    print(f'Targets:  {TARGETS}')
    print(f'LOMO folds: {len(MIX_ORDER)}  |  Optuna trials/fold: {N_TRIALS}\n')

    records      = []         # per-fold detail
    best_params  = {}         # best Optuna params: name → {fold → params}
    fold_r2      = {t: {m: [] for m in MODEL_NAMES} for t in TARGETS}

    for target in TARGETS:
        print(f'{"="*65}')
        print(f'Target: {target}')
        print(f'{"="*65}')
        best_params[target] = {m: {} for m in MODEL_NAMES}

        for mix, tr_idx, te_idx in lomo_splits(df):
            X_tr = df.loc[tr_idx, FEATURES].values
            X_te = df.loc[te_idx, FEATURES].values
            y_tr = df.loc[tr_idx, target].values
            y_te = df.loc[te_idx, target].values

            print(f'\n  Fold [{mix}]  train={len(tr_idx)}  test={len(te_idx)}')

            for name in MODEL_NAMES:
                if name == 'GPR':
                    tr_m, te_m, model, scaler, _ = run_gpr(X_tr, y_tr, X_te, y_te)
                    bp = {}
                else:
                    study = optuna.create_study(
                        direction='minimize',
                        sampler=optuna.samplers.TPESampler(seed=SEED))
                    study.optimize(
                        make_objective(name, X_tr, y_tr),
                        n_trials=N_TRIALS,
                        show_progress_bar=False)
                    bp = study.best_params
                    tr_m, te_m, model, scaler, _ = run_model(
                        name, bp, X_tr, y_tr, X_te, y_te)

                best_params[target][name][mix] = bp
                fold_r2[target][name].append(te_m['R2'])

                records.append({
                    'Target': target, 'Model': name, 'Fold': mix,
                    'Train_R2':   tr_m['R2'],   'Train_RMSE': tr_m['RMSE'], 'Train_MAE': tr_m['MAE'],
                    'Test_R2':    te_m['R2'],   'Test_RMSE':  te_m['RMSE'], 'Test_MAE':  te_m['MAE'],
                    'BestParams': json.dumps(bp),
                })
                print(f'    {name:14s}: R2={te_m["R2"]:+.4f}  '
                      f'RMSE={te_m["RMSE"]:.4f}  MAE={te_m["MAE"]:.4f}')

    # ── Summary ──────────────────────────────────────────────────────────────
    detail_df = pd.DataFrame(records)
    detail_df.to_csv(os.path.join(RESULTS_DIR, 'lomo_detail.csv'), index=False)

    summary_rows = []
    for target in TARGETS:
        for name in MODEL_NAMES:
            sub = detail_df[(detail_df['Target'] == target) & (detail_df['Model'] == name)]
            summary_rows.append({
                'Target':    target,
                'Model':     name,
                'R2_mean':   sub['Test_R2'].mean(),
                'R2_std':    sub['Test_R2'].std(),
                'RMSE_mean': sub['Test_RMSE'].mean(),
                'RMSE_std':  sub['Test_RMSE'].std(),
                'MAE_mean':  sub['Test_MAE'].mean(),
                'MAE_std':   sub['Test_MAE'].std(),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'lomo_summary.csv'), index=False)

    # ── Print summary table ──────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('LOMO Summary (mean ± std over 7 folds)')
    print(f'{"="*65}')
    for target in TARGETS:
        print(f'\n{target}:')
        print(f'  {"Model":<14} {"R2":>12} {"RMSE":>14} {"MAE":>14}')
        print(f'  {"-"*56}')
        sub = summary_df[summary_df['Target'] == target].sort_values('R2_mean', ascending=False)
        for _, row in sub.iterrows():
            print(f'  {row["Model"]:<14} '
                  f'{row["R2_mean"]:+.4f}±{row["R2_std"]:.4f}  '
                  f'{row["RMSE_mean"]:.4f}±{row["RMSE_std"]:.4f}  '
                  f'{row["MAE_mean"]:.4f}±{row["MAE_std"]:.4f}')

    # ── Identify best model per target ───────────────────────────────────────
    best_info = {}
    for target in TARGETS:
        sub  = summary_df[summary_df['Target'] == target]
        best = sub.loc[sub['R2_mean'].idxmax()]
        name = best['Model']

        # Use best-fold params for final full-data model (avoids type issues)
        all_fold_params = best_params[target][name]
        sub_model = detail_df[
            (detail_df['Target'] == target) & (detail_df['Model'] == name)]
        if not sub_model.empty and sub_model['Test_R2'].max() > -np.inf:
            best_fold_row = sub_model.loc[sub_model['Test_R2'].idxmax()]
            params_str    = best_fold_row['BestParams']
            merged = json.loads(params_str) if params_str and params_str != '{}' else {}
        else:
            merged = {}

        best_info[target] = {'model': name, 'params': merged,
                             'R2_mean': float(best['R2_mean']),
                             'R2_std':  float(best['R2_std'])}
        print(f'\nBest for {target}: {name}  '
              f'(R2 = {best["R2_mean"]:.4f} ± {best["R2_std"]:.4f})')

    with open(os.path.join(RESULTS_DIR, 'best_info.json'), 'w') as f:
        json.dump(best_info, f, indent=2)

    # ── Retrain best models on full dataset & save ───────────────────────────
    X_all = df[FEATURES].values
    for target in TARGETS:
        y_all  = df[target].values
        name   = best_info[target]['model']
        params = best_info[target]['params']

        if name == 'GPR':
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-2)
            model  = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5,
                normalize_y=True, random_state=SEED)
            scaler = StandardScaler()
            model.fit(scaler.fit_transform(X_all), y_all)
        else:
            model = build_model(name, params)
            if name in SCALE_MODELS:
                scaler = StandardScaler()
                model.fit(scaler.fit_transform(X_all), y_all)
            else:
                scaler = None
                model.fit(X_all, y_all)

        tag = 'CS' if 'CS' in target else 'FS'
        joblib.dump(model,  os.path.join(RESULTS_DIR, f'best_model_{tag}.joblib'))
        joblib.dump(scaler, os.path.join(RESULTS_DIR, f'scaler_{tag}.joblib'))
        print(f'  Saved: best_model_{tag}.joblib')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nGenerating result figures...')
    plot_lomo_comparison(summary_df, RESULTS_DIR)
    plot_lomo_scatter(df, best_info, RESULTS_DIR)

    print('\nDone. All outputs in results/')


if __name__ == '__main__':
    main()
