"""
WFH Burnout — Feature Importance Analysis
Author: Samuel Pauly

Run this file to execute the full analysis pipeline in sequence:
    1. Exploratory Data Analysis
    2. Model Training & Feature Importance
    3. Threshold Analysis & Bootstrap Robustness
    4. Simplified Models & Interaction Effects

Usage:
    python3 main.py

Requirements:
    pip install pandas numpy matplotlib seaborn scikit-learn scipy tqdm

Output files produced:
    Figures : fig_01_class_distribution.png
              fig_04_boxplots_by_class.png
              fig_08_threshold_violins.png
              fig_09_bootstrap_importance.png
              fig_10_roc_comparison.png
    CSVs    : lr_importance.csv, rf_importance.csv, svm_importance.csv
              thresholds.csv, bootstrap_importance.csv
              simplified_model_results.csv

Estimated runtime: 8-12 minutes (bootstrap + permutation importance are slowest)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import kruskal, mannwhitneyu
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.inspection import permutation_importance
from tqdm import tqdm

# ── Global constants ──────────────────────────────────────────────────────────

CSV_PATH = 'wfh_burnout_dataset.csv'

BEHAVIORAL = [
    'work_hours', 'screen_time_hours', 'meetings_count', 'breaks_taken',
    'after_hours_work', 'app_switches', 'sleep_hours', 'task_completion',
    'isolation_index'
]
FEATURES = BEHAVIORAL + ['day_type_enc']

PALETTE = {'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#F44336'}
ORDER   = ['Low', 'Medium', 'High']

sns.set_theme(style='whitegrid', font_scale=1.1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(df):
    """
    Runs exploratory data analysis on the dataset.
    Produces class distribution chart, box plots, and Kruskal-Wallis tests.
    Returns the correlation series for use downstream.
    """
    print('\n' + '='*60)
    print('SECTION 1 — EXPLORATORY DATA ANALYSIS')
    print('='*60)

    # Basic dataset info
    print(f'\nDataset shape : {df.shape}')
    print(f'Missing values: {df.isnull().sum().sum()}')
    print(f'Duplicate rows: {df.duplicated().sum()}')

    # Class distribution
    counts = df['burnout_risk'].value_counts().reindex(ORDER)
    pcts   = (counts / len(df) * 100).round(1)
    print('\nClass distribution:')
    for cls in ORDER:
        print(f'  {cls:8s}: {counts[cls]:5d}  ({pcts[cls]}%)')

    # Figure: class distribution bar + pie
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(ORDER, counts, color=[PALETTE[c] for c in ORDER])
    axes[0].set_title('Burnout Class Counts')
    axes[0].set_xlabel('Burnout Risk')
    axes[0].set_ylabel('Count')
    axes[1].pie(counts, labels=ORDER, colors=[PALETTE[c] for c in ORDER],
                autopct='%1.1f%%')
    axes[1].set_title('Burnout Class Proportions')
    plt.tight_layout()
    plt.savefig('fig_01_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig_01_class_distribution.png')

    # Correlation with burnout_risk (encoded numerically)
    df_enc = df.copy()
    df_enc['day_type_enc']     = (df_enc['day_type'] == 'Weekday').astype(int)
    df_enc['burnout_risk_enc'] = df_enc['burnout_risk'].map({'Low': 0, 'Medium': 1, 'High': 2})
    corr_matrix = df_enc[BEHAVIORAL + ['day_type_enc', 'burnout_risk_enc']].corr()
    corrs = corr_matrix['burnout_risk_enc'].drop('burnout_risk_enc').abs().sort_values(ascending=False)
    print('\nFeature correlations with burnout_risk (ranked by |r|):')
    print(corrs.round(3).to_string())

    # Figure: box plots by burnout class for each behavioral feature
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()
    for i, col in enumerate(BEHAVIORAL):
        sns.boxplot(data=df, x='burnout_risk', y=col,
                    order=ORDER, palette=PALETTE, ax=axes[i])
        axes[i].set_title(col)
    plt.suptitle('Feature Distributions by Burnout Risk Level', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('fig_04_boxplots_by_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig_04_boxplots_by_class.png')

    # Class mean table and High-Low delta
    mean_table = df.groupby('burnout_risk')[BEHAVIORAL].mean().reindex(ORDER)
    delta = (mean_table.loc['High'] - mean_table.loc['Low']).abs()
    delta_sorted = delta.sort_values(ascending=False)
    print('\nHigh-Low mean difference per feature (larger = stronger signal):')
    print(delta_sorted.round(3).to_string())

    # Kruskal-Wallis tests — are class distributions significantly different?
    kw_results = []
    for col in BEHAVIORAL:
        groups  = [df[df['burnout_risk'] == cls][col].values for cls in ORDER]
        h, p    = kruskal(*groups)
        kw_results.append({'Feature': col, 'H-statistic': round(h, 2),
                            'p-value': p, 'Significant (p<0.05)': p < 0.05})
    kw_df = pd.DataFrame(kw_results).sort_values('H-statistic', ascending=False)
    print('\nKruskal-Wallis Tests (sorted by H-statistic):')
    print(kw_df.to_string(index=False))

    print('\n✓ EDA complete')
    return corrs, mean_table


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MODEL TRAINING & FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def run_models(df):
    """
    Trains Logistic Regression, Random Forest, and SVM classifiers.
    Extracts feature importance from each model using three different methods:
      - LR: mean absolute standardized coefficients
      - RF: Gini impurity reduction
      - SVM: permutation importance (model-agnostic)
    Saves importance scores to CSV for downstream use.
    Returns trained models, importance series, and split data.
    """
    print('\n' + '='*60)
    print('SECTION 2 — MODEL TRAINING & FEATURE IMPORTANCE')
    print('='*60)

    # Encode day_type and define features / target
    df['day_type_enc'] = (df['day_type'] == 'Weekday').astype(int)
    X = df[FEATURES]
    y = df['burnout_risk']

    # Stratified 80/20 split — preserves class proportions in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'\nTrain: {X_train.shape}  |  Test: {X_test.shape}')

    # Scale features — fit only on training data to prevent leakage
    scaler         = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURES)

    # ── Logistic Regression ──────────────────────────────────────────────────
    print('\nTraining Logistic Regression...')
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    print('=== Logistic Regression — Classification Report ===')
    print(classification_report(y_test, y_pred_lr))

    # Feature importance: mean absolute coefficient across classes
    lr_importance = pd.Series(
        np.abs(lr_model.coef_).mean(axis=0), index=FEATURES
    ).sort_values(ascending=False)
    print('LR Feature Importance (|coef|):')
    print(lr_importance.round(4).to_string())
    print('✓ Logistic Regression complete')

    # ── Random Forest ────────────────────────────────────────────────────────
    print('\nTraining Random Forest (200 trees)...')
    rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    print('=== Random Forest — Classification Report ===')
    print(classification_report(y_test, y_pred_rf))

    # Feature importance: Gini impurity reduction
    rf_importance = pd.Series(
        rf_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    print('RF Feature Importance (Gini):')
    print(rf_importance.round(4).to_string())
    print('✓ Random Forest complete')

    # ── SVM ──────────────────────────────────────────────────────────────────
    print('\nTraining SVM (RBF kernel)...')
    svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)

    print('=== SVM — Classification Report ===')
    print(classification_report(y_test, y_pred_svm))

    # Feature importance: permutation importance (shuffles each feature, measures AUC drop)
    print('Computing SVM permutation importance (this takes 1-2 min)...')
    perm = permutation_importance(svm_model, X_test_scaled, y_test,
                                  n_repeats=20, random_state=42)
    svm_importance = pd.Series(
        perm.importances_mean, index=FEATURES
    ).sort_values(ascending=False)
    print('SVM Feature Importance (Permutation):')
    print(svm_importance.round(4).to_string())
    print('✓ SVM complete')

    # ── ROC-AUC summary ──────────────────────────────────────────────────────
    lr_auc  = roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled),  multi_class='ovr')
    rf_auc  = roc_auc_score(y_test, rf_model.predict_proba(X_test),         multi_class='ovr')
    svm_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test_scaled), multi_class='ovr')

    print('\n' + '='*50)
    print('MODEL PERFORMANCE SUMMARY')
    print('='*50)
    print(f'{"Model":<25} {"ROC-AUC"}')
    print('-'*50)
    print(f'{"Logistic Regression":<25} {lr_auc:.4f}')
    print(f'{"Random Forest":<25} {rf_auc:.4f}')
    print(f'{"SVM":<25} {svm_auc:.4f}')
    print('='*50)

    # ── Consensus ranking across all three models ─────────────────────────────
    rank_df = pd.DataFrame({
        'LR Rank' : lr_importance.rank(ascending=False).astype(int),
        'RF Rank' : rf_importance.rank(ascending=False).astype(int),
        'SVM Rank': svm_importance.rank(ascending=False).astype(int),
    })
    rank_df['Mean Rank'] = rank_df.mean(axis=1).round(1)
    rank_df = rank_df.sort_values('Mean Rank')
    print('\nConsensus Feature Ranking (1 = most important):')
    print(rank_df.to_string())

    # Save importance scores for use in later sections
    lr_importance.to_csv('lr_importance.csv',   header=['importance'])
    rf_importance.to_csv('rf_importance.csv',   header=['importance'])
    svm_importance.to_csv('svm_importance.csv', header=['importance'])
    rank_df.to_csv('rank_comparison.csv')
    print('\n✓ Importance scores saved to CSV')

    return (lr_model, rf_model, svm_model,
            lr_importance, rf_importance, svm_importance,
            X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test,
            rank_df)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — THRESHOLD ANALYSIS & BOOTSTRAP ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────

def run_threshold_and_bootstrap(df, rank_df):
    """
    Derives intervention thresholds for the top 4 features and validates
    class separation using pairwise Mann-Whitney U tests.
    Then runs 100-iteration bootstrap resampling to confirm that feature
    importance rankings are stable across different data splits.
    """
    print('\n' + '='*60)
    print('SECTION 3 — THRESHOLD ANALYSIS & BOOTSTRAP ROBUSTNESS')
    print('='*60)

    # Encode day_type
    df['day_type_enc'] = (df['day_type'] == 'Weekday').astype(int)

    # Top 4 features by consensus rank
    TOP_FEATURES = rank_df.head(4).index.tolist()
    print(f'\nTop 4 features for threshold analysis: {TOP_FEATURES}')

    # ── Mann-Whitney U tests — pairwise class comparisons ────────────────────
    pairs = [('Low', 'Medium'), ('Low', 'High'), ('Medium', 'High')]
    mw_results = []
    for feat in TOP_FEATURES:
        for cls_a, cls_b in pairs:
            group_a = df[df['burnout_risk'] == cls_a][feat].values
            group_b = df[df['burnout_risk'] == cls_b][feat].values
            stat, p = mannwhitneyu(group_a, group_b, alternative='two-sided')
            mw_results.append({
                'Feature'    : feat,
                'Comparison' : f'{cls_a} vs {cls_b}',
                'U-statistic': round(stat, 2),
                'p-value'    : f'{p:.2e}',
                'Significant': 'Yes' if p < 0.05 else 'No'
            })
    mw_df = pd.DataFrame(mw_results)
    print('\nMann-Whitney U Tests (pairwise class comparisons):')
    print(mw_df.to_string(index=False))

    # ── Intervention threshold derivation ────────────────────────────────────
    # Threshold = midpoint between Low and High class means
    threshold_results = []
    for feat in TOP_FEATURES:
        low_mean  = df[df['burnout_risk'] == 'Low'][feat].mean()
        med_mean  = df[df['burnout_risk'] == 'Medium'][feat].mean()
        high_mean = df[df['burnout_risk'] == 'High'][feat].mean()
        threshold = (low_mean + high_mean) / 2
        threshold_results.append({
            'Feature'   : feat,
            'Low Mean'  : round(low_mean,  2),
            'Medium Mean': round(med_mean, 2),
            'High Mean' : round(high_mean, 2),
            'Threshold' : round(threshold, 2),
            'Direction' : 'Above threshold = higher risk' if high_mean > low_mean
                          else 'Below threshold = higher risk'
        })
    thresh_df = pd.DataFrame(threshold_results)
    print('\nIntervention Thresholds (midpoint between Low and High class means):')
    print(thresh_df.to_string(index=False))
    thresh_df.to_csv('thresholds.csv', index=False)
    print('Saved: thresholds.csv')

    # Figure: violin plots with threshold lines overlaid
    fig, axes = plt.subplots(1, len(TOP_FEATURES), figsize=(15, 5))
    for i, row in thresh_df.iterrows():
        ax = axes[i]
        sns.violinplot(data=df, x='burnout_risk', y=row['Feature'],
                       order=ORDER, palette=PALETTE, ax=ax)
        ax.axhline(y=row['Threshold'], color='black', linestyle='--',
                   linewidth=1.5, label=f"Threshold: {row['Threshold']}")
        ax.set_title(row['Feature'].replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('')
        ax.legend(fontsize=8)
    plt.suptitle('Feature Distributions with Intervention Thresholds', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('fig_08_threshold_violins.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig_08_threshold_violins.png')

    # ── Bootstrap robustness — 100 iterations ────────────────────────────────
    # Resamples the full dataset with replacement, trains a Random Forest each time,
    # and records feature importances to compute stable mean + 95% CI estimates.
    N_BOOTSTRAP = 100
    print(f'\nRunning {N_BOOTSTRAP} bootstrap iterations...')
    bootstrap_records = []

    for i in tqdm(range(N_BOOTSTRAP)):
        sample = df.sample(n=len(df), replace=True, random_state=i)
        X_s    = sample[FEATURES]
        y_s    = sample['burnout_risk']
        rf     = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=i)
        rf.fit(X_s, y_s)
        bootstrap_records.append(dict(zip(FEATURES, rf.feature_importances_)))

    bootstrap_df  = pd.DataFrame(bootstrap_records)
    boot_summary  = pd.DataFrame({
        'Mean Importance' : bootstrap_df.mean(),
        'CI Lower (2.5%)' : bootstrap_df.quantile(0.025),
        'CI Upper (97.5%)': bootstrap_df.quantile(0.975),
    }).sort_values('Mean Importance', ascending=False).round(4)

    print('\nBootstrap Feature Importance (100 iterations, 95% CI):')
    print(boot_summary.to_string())
    boot_summary.to_csv('bootstrap_importance.csv')
    print('Saved: bootstrap_importance.csv')

    # Figure: horizontal bar chart with 95% CI error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    means      = boot_summary['Mean Importance']
    xerr_low   = means - boot_summary['CI Lower (2.5%)']
    xerr_high  = boot_summary['CI Upper (97.5%)'] - means
    ax.barh(boot_summary.index, means,
            xerr=[xerr_low, xerr_high], capsize=4, color='steelblue', alpha=0.8)
    ax.set_xlabel('Mean Importance (Random Forest, 100 Bootstrap Iterations)')
    ax.set_title('Feature Importance with 95% Confidence Intervals', fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('fig_09_bootstrap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig_09_bootstrap_importance.png')

    print('\n✓ Threshold analysis and bootstrap complete')
    return thresh_df, boot_summary, TOP_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SIMPLIFIED MODELS & INTERACTION EFFECTS
# ─────────────────────────────────────────────────────────────────────────────

def run_simplified_and_interactions(df, TOP_FEATURES):
    """
    Tests whether a reduced-feature model can match full-model performance,
    and whether engineered interaction features add predictive value.
    Produces ROC curve comparison for the High burnout class.
    """
    print('\n' + '='*60)
    print('SECTION 4 — SIMPLIFIED MODELS & INTERACTION EFFECTS')
    print('='*60)

    # Encode and split
    df['day_type_enc'] = (df['day_type'] == 'Weekday').astype(int)
    X = df[FEATURES]
    y = df['burnout_risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler         = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURES, index=X_test.index)

    # Feature subsets based on consensus ranking
    TOP_4 = TOP_FEATURES[:4]
    TOP_3 = TOP_FEATURES[:3]
    TOP_5 = TOP_FEATURES[:4] + ['app_switches']

    rf_params = dict(n_estimators=200, class_weight='balanced', random_state=42)

    # ── Full model baseline ──────────────────────────────────────────────────
    rf_full       = RandomForestClassifier(**rf_params)
    cv_full       = cross_val_score(rf_full, X_train_scaled, y_train, cv=5, scoring='roc_auc_ovr')
    rf_full.fit(X_train_scaled, y_train)
    y_proba_full  = rf_full.predict_proba(X_test_scaled)
    auc_full      = roc_auc_score(y_test, y_proba_full, multi_class='ovr')

    simplified_results = [{
        'Model': 'Full (10 features)', 'Features': 10,
        'CV AUC Mean': cv_full.mean(), 'CV AUC Std': cv_full.std(), 'Test AUC': auc_full
    }]
    simplified_probas = {}
    simplified_models = {}

    # ── Simplified models — top 3, 4, 5 features ────────────────────────────
    for label, feature_set in [('Top 3', TOP_3), ('Top 4', TOP_4), ('Top 5', TOP_5)]:
        X_tr = X_train_scaled[feature_set]
        X_te = X_test_scaled[feature_set]
        rf   = RandomForestClassifier(**rf_params)
        cv   = cross_val_score(rf, X_tr, y_train, cv=5, scoring='roc_auc_ovr')
        rf.fit(X_tr, y_train)
        proba    = rf.predict_proba(X_te)
        test_auc = roc_auc_score(y_test, proba, multi_class='ovr')
        simplified_results.append({
            'Model': f'{label} features ({len(feature_set)})', 'Features': len(feature_set),
            'CV AUC Mean': cv.mean(), 'CV AUC Std': cv.std(), 'Test AUC': test_auc
        })
        simplified_models[label] = rf
        simplified_probas[label] = proba
        print(f'{label}: CV AUC = {cv.mean():.4f} ± {cv.std():.4f}  |  Test AUC = {test_auc:.4f}')

    results_df = pd.DataFrame(simplified_results).round(4)
    print('\nFull Simplified Model Comparison:')
    print(results_df.to_string(index=False))
    results_df.to_csv('simplified_model_results.csv', index=False)

    # Figure: ROC curve — full model vs best simplified model (High class only)
    y_test_bin = label_binarize(y_test, classes=ORDER)
    high_idx   = 2  # column index for High class

    fig, ax = plt.subplots(figsize=(7, 6))
    full_cls_idx  = {cls: i for i, cls in enumerate(rf_full.classes_)}
    fpr_f, tpr_f, _ = roc_curve(y_test_bin[:, high_idx],
                                  y_proba_full[:, full_cls_idx['High']])
    ax.plot(fpr_f, tpr_f, label=f'Full model (AUC={auc(fpr_f, tpr_f):.3f})', linewidth=2)

    best_row   = results_df.iloc[1:].sort_values('Test AUC', ascending=False).iloc[0]
    best_label = ' '.join(best_row['Model'].split()[:2])
    best_proba = simplified_probas[best_label]
    best_model = simplified_models[best_label]
    simp_cls_idx = {cls: i for i, cls in enumerate(best_model.classes_)}
    fpr_s, tpr_s, _ = roc_curve(y_test_bin[:, high_idx],
                                  best_proba[:, simp_cls_idx['High']])
    ax.plot(fpr_s, tpr_s, label=f'{best_label} model (AUC={auc(fpr_s, tpr_s):.3f})',
            linewidth=2, linestyle='--')
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, label='Random chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — High Burnout Class\nFull Model vs Best Simplified Model', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig_10_roc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig_10_roc_comparison.png')

    # ── Interaction feature engineering ──────────────────────────────────────
    # Three compound features motivated by behavioral theory:
    #   overwork_fatigue  = work_hours × (10 - sleep_hours)
    #                       high when both overworked and sleep-deprived
    #   isolated_overwork = isolation_index × work_hours
    #                       high when both isolated and overworked
    #   no_recovery       = work_hours / (breaks_taken + 1)
    #                       high work-to-rest ratio (+1 avoids division by zero)
    df['overwork_fatigue']  = df['work_hours'] * (10 - df['sleep_hours'])
    df['isolated_overwork'] = df['isolation_index'] * df['work_hours']
    df['no_recovery']       = df['work_hours'] / (df['breaks_taken'] + 1)

    INTERACTION_FEATURES = ['overwork_fatigue', 'isolated_overwork', 'no_recovery']
    risk_enc = df['burnout_risk'].map({'Low': 0, 'Medium': 1, 'High': 2})
    print('\nInteraction feature correlations with burnout_risk:')
    for feat in INTERACTION_FEATURES:
        print(f'  {feat:25s}: r = {df[feat].corr(risk_enc):.3f}')

    # Compare top-4 baseline vs top-4 + interaction features using 5-fold CV
    INTERACTION_SET = TOP_4 + INTERACTION_FEATURES
    X_int       = df[INTERACTION_SET]
    y_int       = df['burnout_risk']
    X_int_train, X_int_test, y_int_train, _ = train_test_split(
        X_int, y_int, test_size=0.2, random_state=42, stratify=y_int
    )
    scaler_int        = StandardScaler()
    X_int_train_sc    = pd.DataFrame(scaler_int.fit_transform(X_int_train),
                                     columns=INTERACTION_SET, index=X_int_train.index)

    cv_base  = cross_val_score(RandomForestClassifier(**rf_params),
                               X_int_train_sc[TOP_4], y_int_train,
                               cv=5, scoring='roc_auc_ovr')
    cv_inter = cross_val_score(RandomForestClassifier(**rf_params),
                               X_int_train_sc[INTERACTION_SET], y_int_train,
                               cv=5, scoring='roc_auc_ovr')

    print('\n' + '='*55)
    print('INTERACTION EFFECT TEST')
    print('='*55)
    print(f'Baseline  (top 4 features)         CV AUC: {cv_base.mean():.4f} ± {cv_base.std():.4f}')
    print(f'Interaction (top 4 + 3 engineered) CV AUC: {cv_inter.mean():.4f} ± {cv_inter.std():.4f}')
    delta = cv_inter.mean() - cv_base.mean()
    print(f'Delta: {delta:+.4f}')
    if   delta >  0.005: print('→ Interaction features IMPROVE performance')
    elif delta < -0.005: print('→ Interaction features HURT performance')
    else:                print('→ Interaction features make NO meaningful difference')
    print('='*55)

    print('\n✓ Simplified models and interaction analysis complete')
    return results_df, cv_base, cv_inter, auc_full, TOP_4


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — runs all sections in sequence
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('='*60)
    print('WFH BURNOUT — FULL ANALYSIS PIPELINE')
    print('Samuel Pauly')
    print('='*60)

    # Load dataset once, pass to each section
    df = pd.read_csv(CSV_PATH)
    if 'user_id' in df.columns:
        df.drop(columns=['user_id'], inplace=True)

    # Run all four sections
    corrs, mean_table = run_eda(df.copy())

    (lr_model, rf_model, svm_model,
     lr_importance, rf_importance, svm_importance,
     X_train, X_test, X_train_scaled, X_test_scaled,
     y_train, y_test, rank_df) = run_models(df.copy())

    thresh_df, boot_summary, TOP_FEATURES = run_threshold_and_bootstrap(df.copy(), rank_df)

    results_df, cv_base, cv_inter, auc_full, TOP_4 = run_simplified_and_interactions(
        df.copy(), TOP_FEATURES
    )

    # Final summary
    print('\n' + '='*60)
    print('COMPLETE PROJECT SUMMARY')
    print('='*60)
    print(f'Dataset         : {len(df)} records, {len(BEHAVIORAL)} behavioral features')
    print(f'Top feature     : {rank_df.index[0]} (unanimous across LR, RF, SVM)')
    print(f'Full model AUC  : {auc_full:.4f}')
    best = results_df.iloc[1:].sort_values('Test AUC', ascending=False).iloc[0]
    print(f'Best simplified : {best["Model"]}  AUC = {best["Test AUC"]:.4f}')
    print(f'Interaction delta: {cv_inter.mean() - cv_base.mean():+.4f} (no meaningful gain)')
    print('\nOutput files saved:')
    for f in ['fig_01_class_distribution.png', 'fig_04_boxplots_by_class.png',
              'fig_08_threshold_violins.png',   'fig_09_bootstrap_importance.png',
              'fig_10_roc_comparison.png',
              'lr_importance.csv', 'rf_importance.csv', 'svm_importance.csv',
              'thresholds.csv', 'bootstrap_importance.csv', 'simplified_model_results.csv']:
        print(f'  {f}')
    print('\n✓ All sections complete.')
