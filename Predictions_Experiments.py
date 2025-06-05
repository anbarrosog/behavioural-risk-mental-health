#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")


# In[27]:


# --- LIBRARIES ---
import os
import warnings
import contextlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[28]:


# --- SETTINGS ---
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['XGBOOST_DISABLE_WARNINGS'] = '1'


# In[29]:


# --- LOAD DATA (Binary and Weighted Items Datasets) ---
import pandas as pd

paths = {
    'features_binary': 'binary_items_dataset.xlsx',       # Binary feature matrix
    'features_weighted': 'weighted_items_dataset.xlsx',   # Weighted feature matrix
    'sociodemo': 'sociodemographic_data.xlsx',            # Sociodemographic data
    'risk': 'turnos_completos_unicos_risk.xlsx'           # Risk outcome labels
}

# Load the binary dataset
features_binary = pd.read_excel(paths['features_binary'])
# Load the weighted dataset
features_weighted = pd.read_excel(paths['features_weighted'])
# Load sociodemographic and risk data
sociodemo = pd.read_excel(paths['sociodemo'])
risk = pd.read_excel(paths['risk'])


# In[30]:


# --- Function to preprocess any features dataset ---
def preprocess_features(features, sociodemo, risk):
    # Merge features with sociodemographic and risk data
    df = features.merge(sociodemo, on='NHC', how='left')
    df = df.merge(risk[['Fecha', 'Turno', 'Aggressive', 'Self-Harm', 'Absconding']],
                  on=['Fecha', 'Turno'], how='left')
    
    # Assign binary and multiclass risk labels
    def assign_multiclass(row):
        if pd.notna(row['Aggressive']): return 'Aggressive'
        if pd.notna(row['Self-Harm']): return 'Self-Harm'
        if pd.notna(row['Absconding']): return 'Absconding'
        return 'No_risk'

    def assign_binary(row):
        if pd.notna(row['Aggressive']) or pd.notna(row['Self-Harm']) or pd.notna(row['Absconding']):
            return 'Risk'
        return 'No_Risk'
    
    df['Risk_Multiclass'] = df.apply(assign_multiclass, axis=1)
    df['Risk_Binary'] = df.apply(assign_binary, axis=1)
    df = df[df['Risk_Binary'].notna()].copy()
    
    # Feature preprocessing
    drop_cols = ['NHC', 'Fecha', 'Turno', 'Aggressive', 'Self-Harm', 'Absconding', 'Risk_Multiclass', 'Risk_Binary']
    X = df.drop(columns=drop_cols)
    y = df['Risk_Binary'].map({'No_Risk': 0, 'Risk': 1})
    groups = df['NHC']
    
    # Dummify categorical variables, impute missing values, scale features
    X = pd.get_dummies(X, drop_first=True)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Data overview
    print("Number of samples:", X_scaled.shape[0])
    print("Number of variables after dummies:", X_scaled.shape[1])
    print("Binary classes (0=No_Risk, 1=Risk):")
    print(y.value_counts())
    print("\nSample of labeled data:")
    display(df[['NHC', 'Fecha', 'Turno', 'Risk_Binary']].sample(5, random_state=42))
    
    return df, X_scaled, y, groups

# --- Process both datasets ---

print("\n--- Processing BINARY ITEMS DATASET ---")
df_binary, X_scaled_binary, y_binary, groups_binary = preprocess_features(
    features_binary, sociodemo, risk
)

print("\n--- Processing WEIGHTED ITEMS DATASET ---")
df_weighted, X_scaled_weighted, y_weighted, groups_weighted = preprocess_features(
    features_weighted, sociodemo, risk
)


# In[31]:


### LOGISTIC REGRESSION – LOPOCV (BY DAY AND BY SHIFT, BINARY & MULTICLASS, BOTH DATASETS)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ---------- HELPERS ----------
def collect_metrics_bin(y_true, y_pred, y_score):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_score)
    }

def collect_metrics_mc(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        # AUC para multiclass requiere one-vs-rest y más código; dime si lo quieres
    }

# ---------- BINARY CLASSIFICATION ----------
# Loop: group_var = "turn" or "day", dataset = binary or weighted
results_logreg_bin = {}  # For all metrics and arrays

for group_label, group_field in [('Turn', 'NHC'), ('Day', 'Fecha')]:
    for ds_label, (df, X_scaled, y, groups) in [
        ('Binary', (df_binary, X_scaled_binary, y_binary, df_binary)),
        ('Weighted', (df_weighted, X_scaled_weighted, y_weighted, df_weighted))
    ]:
        logo = LeaveOneGroupOut()
        model = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='auto', solver='lbfgs')
        y_true, y_pred, y_score = [], [], []
        # Choose grouping
        if group_field == 'NHC':
            group_vals = df['NHC']
        else:
            group_vals = df['Fecha']

        for train_idx, test_idx in logo.split(X_scaled, y, group_vals):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]
            model.fit(X_train, y_train)
            y_pred_fold = model.predict(X_test)
            y_score_fold = model.predict_proba(X_test)[:, 1]
            y_true.extend(y_test)
            y_pred.extend(y_pred_fold)
            y_score.extend(y_score_fold)
        # Save results
        key = f"{ds_label}_Bin_LOPOCV_{group_label}"
        results_logreg_bin[key] = {
            'y_true': np.array(y_true),
            'y_pred': np.array(y_pred),
            'y_score': np.array(y_score),
            'metrics': collect_metrics_bin(y_true, y_pred, y_score)
        }
        # Print summary metrics for this experiment
        print(f"\nLOGISTIC REGRESSION | {ds_label} dataset | BINARY | LOPOCV by {group_label}")
        for metric, value in results_logreg_bin[key]['metrics'].items():
            print(f"{metric}: {value:.3f}")

# ---------- MULTICLASS CLASSIFICATION ----------
results_logreg_mc = {}

for group_label, group_field in [('Turn', 'NHC'), ('Day', 'Fecha')]:
    for ds_label, (df, X_scaled) in [
        ('Binary', (df_binary, X_scaled_binary)),
        ('Weighted', (df_weighted, X_scaled_weighted))
    ]:
        # Prepare multiclass
        df_mc = df[df['Risk_Multiclass'].notna()].copy()
        drop_cols = ['NHC', 'Fecha', 'Turno', 'Aggressive', 'Self-Harm', 'Absconding', 'Risk_Multiclass', 'Risk_Binary']
        X_mc = df_mc.drop(columns=drop_cols)
        X_mc = pd.get_dummies(X_mc, drop_first=True)
        y_mc = LabelEncoder().fit_transform(df_mc['Risk_Multiclass'])
        labels_map = dict(enumerate(LabelEncoder().fit(df_mc['Risk_Multiclass']).classes_))
        # Align groups
        if group_field == 'NHC':
            group_vals = df_mc['NHC']
        else:
            group_vals = df_mc['Fecha']
        logo = LeaveOneGroupOut()
        model = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='multinomial', solver='lbfgs')
        y_true_mc, y_pred_mc = [], []
        for train_idx, test_idx in logo.split(X_mc, y_mc, group_vals):
            X_train, X_test = X_mc.values[train_idx], X_mc.values[test_idx]
            y_train, y_test = y_mc[train_idx], y_mc[test_idx]
            model.fit(X_train, y_train)
            y_pred_fold = model.predict(X_test)
            y_true_mc.extend(y_test)
            y_pred_mc.extend(y_pred_fold)
        # Save results
        key = f"{ds_label}_MC_LOPOCV_{group_label}"
        results_logreg_mc[key] = {
            'y_true': np.array(y_true_mc),
            'y_pred': np.array(y_pred_mc),
            'metrics': collect_metrics_mc(y_true_mc, y_pred_mc),
            'labels_map': labels_map
        }
        print(f"\nLOGISTIC REGRESSION | {ds_label} dataset | MULTICLASS | LOPOCV by {group_label}")
        for metric, value in results_logreg_mc[key]['metrics'].items():
            print(f"{metric}: {value:.3f}")


# In[32]:


#### RANDOM FOREST – LOPOCV (BY DAY AND BY SHIFT, BINARY & MULTICLASS, BOTH DATASETS)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------- BINARY CLASSIFICATION ----------
results_rf_bin = {}

for group_label, group_field in [('Turn', 'NHC'), ('Day', 'Fecha')]:
    for ds_label, (df, X_scaled, y, groups) in [
        ('Binary', (df_binary, X_scaled_binary, y_binary, df_binary)),
        ('Weighted', (df_weighted, X_scaled_weighted, y_weighted, df_weighted))
    ]:
        logo = LeaveOneGroupOut()
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        y_true, y_pred, y_score = [], [], []
        # Choose grouping
        if group_field == 'NHC':
            group_vals = df['NHC']
        else:
            group_vals = df['Fecha']

        for train_idx, test_idx in logo.split(X_scaled, y, group_vals):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]
            model.fit(X_train, y_train)
            y_pred_fold = model.predict(X_test)
            y_score_fold = model.predict_proba(X_test)[:, 1]
            y_true.extend(y_test)
            y_pred.extend(y_pred_fold)
            y_score.extend(y_score_fold)
        key = f"{ds_label}_Bin_LOPOCV_{group_label}"
        results_rf_bin[key] = {
            'y_true': np.array(y_true),
            'y_pred': np.array(y_pred),
            'y_score': np.array(y_score),
            'metrics': collect_metrics_bin(y_true, y_pred, y_score)
        }
        print(f"\nRANDOM FOREST | {ds_label} dataset | BINARY | LOPOCV by {group_label}")
        for metric, value in results_rf_bin[key]['metrics'].items():
            print(f"{metric}: {value:.3f}")

# ---------- MULTICLASS CLASSIFICATION ----------
results_rf_mc = {}

for group_label, group_field in [('Turn', 'NHC'), ('Day', 'Fecha')]:
    for ds_label, (df, X_scaled) in [
        ('Binary', (df_binary, X_scaled_binary)),
        ('Weighted', (df_weighted, X_scaled_weighted))
    ]:
        df_mc = df[df['Risk_Multiclass'].notna()].copy()
        drop_cols = ['NHC', 'Fecha', 'Turno', 'Aggressive', 'Self-Harm', 'Absconding', 'Risk_Multiclass', 'Risk_Binary']
        X_mc = df_mc.drop(columns=drop_cols)
        X_mc = pd.get_dummies(X_mc, drop_first=True)
        y_mc = LabelEncoder().fit_transform(df_mc['Risk_Multiclass'])
        labels_map = dict(enumerate(LabelEncoder().fit(df_mc['Risk_Multiclass']).classes_))
        if group_field == 'NHC':
            group_vals = df_mc['NHC']
        else:
            group_vals = df_mc['Fecha']
        logo = LeaveOneGroupOut()
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        y_true_mc, y_pred_mc = [], []
        for train_idx, test_idx in logo.split(X_mc, y_mc, group_vals):
            X_train, X_test = X_mc.values[train_idx], X_mc.values[test_idx]
            y_train, y_test = y_mc[train_idx], y_mc[test_idx]
            model.fit(X_train, y_train)
            y_pred_fold = model.predict(X_test)
            y_true_mc.extend(y_test)
            y_pred_mc.extend(y_pred_fold)
        key = f"{ds_label}_MC_LOPOCV_{group_label}"
        results_rf_mc[key] = {
            'y_true': np.array(y_true_mc),
            'y_pred': np.array(y_pred_mc),
            'metrics': collect_metrics_mc(y_true_mc, y_pred_mc),
            'labels_map': labels_map
        }
        print(f"\nRANDOM FOREST | {ds_label} dataset | MULTICLASS | LOPOCV by {group_label}")
        for metric, value in results_rf_mc[key]['metrics'].items():
            print(f"{metric}: {value:.3f}")


# In[33]:


### XGBOOST – LOPOCV (BY DAY AND BY SHIFT, BINARY & MULTICLASS, BOTH DATASETS

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# ---------- BINARY CLASSIFICATION ----------
results_xgb_bin = {}

for group_label, group_field in [('Turn', 'NHC'), ('Day', 'Fecha')]:
    for ds_label, (df, X_scaled, y, groups) in [
        ('Binary', (df_binary, X_scaled_binary, y_binary, df_binary)),
        ('Weighted', (df_weighted, X_scaled_weighted, y_weighted, df_weighted))
    ]:
        # Handle class imbalance for XGBoost
        scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
        logo = LeaveOneGroupOut()
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        y_true, y_pred, y_score = [], [], []
        if group_field == 'NHC':
            group_vals = df['NHC']
        else:
            group_vals = df['Fecha']

        for train_idx, test_idx in logo.split(X_scaled, y, group_vals):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]
            model.fit(X_train, y_train)
            y_pred_fold = model.predict(X_test)
            y_score_fold = model.predict_proba(X_test)[:, 1]
            y_true.extend(y_test)
            y_pred.extend(y_pred_fold)
            y_score.extend(y_score_fold)
        key = f"{ds_label}_Bin_LOPOCV_{group_label}"
        results_xgb_bin[key] = {
            'y_true': np.array(y_true),
            'y_pred': np.array(y_pred),
            'y_score': np.array(y_score),
            'metrics': collect_metrics_bin(y_true, y_pred, y_score)
        }
        print(f"\nXGBOOST | {ds_label} dataset | BINARY | LOPOCV by {group_label}")
        for metric, value in results_xgb_bin[key]['metrics'].items():
            print(f"{metric}: {value:.3f}")

# ---------- MULTICLASS CLASSIFICATION ----------
results_xgb_mc = {}

for group_label, group_field in [('Turn', 'NHC'), ('Day', 'Fecha')]:
    for ds_label, (df, X_scaled) in [
        ('Binary', (df_binary, X_scaled_binary)),
        ('Weighted', (df_weighted, X_scaled_weighted))
    ]:
        df_mc = df[df['Risk_Multiclass'].notna()].copy()
        drop_cols = ['NHC', 'Fecha', 'Turno', 'Aggressive', 'Self-Harm', 'Absconding', 'Risk_Multiclass', 'Risk_Binary']
        X_mc = df_mc.drop(columns=drop_cols)
        X_mc = pd.get_dummies(X_mc, drop_first=True)
        y_mc = LabelEncoder().fit_transform(df_mc['Risk_Multiclass'])
        labels_map = dict(enumerate(LabelEncoder().fit(df_mc['Risk_Multiclass']).classes_))
        if group_field == 'NHC':
            group_vals = df_mc['NHC']
        else:
            group_vals = df_mc['Fecha']
        logo = LeaveOneGroupOut()
        model = XGBClassifier(
            objective='multi:softmax',
            num_class=len(labels_map),
            eval_metric='mlogloss',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        y_true_mc, y_pred_mc = [], []
        for train_idx, test_idx in logo.split(X_mc, y_mc, group_vals):
            X_train, X_test = X_mc.values[train_idx], X_mc.values[test_idx]
            y_train, y_test = y_mc[train_idx], y_mc[test_idx]
            model.fit(X_train, y_train)
            y_pred_fold = model.predict(X_test)
            y_true_mc.extend(y_test)
            y_pred_mc.extend(y_pred_fold)
        key = f"{ds_label}_MC_LOPOCV_{group_label}"
        results_xgb_mc[key] = {
            'y_true': np.array(y_true_mc),
            'y_pred': np.array(y_pred_mc),
            'metrics': collect_metrics_mc(y_true_mc, y_pred_mc),
            'labels_map': labels_map
        }
        print(f"\nXGBOOST | {ds_label} dataset | MULTICLASS | LOPOCV by {group_label}")
        for metric, value in results_xgb_mc[key]['metrics'].items():
            print(f"{metric}: {value:.3f}")


# In[34]:


### NEURAL NETWORK (Keras) – LOPOCV (BY DAY AND BY SHIFT, BINARY & MULTICLASS, BOTH DATASETS)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- Helper for binary neural net ---
def create_binary_nn(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Helper for multiclass neural net ---
def create_multiclass_nn(input_dim, output_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------- BINARY CLASSIFICATION ----------
results_nn_bin = {}

for group_label, group_field in [('Turn', 'NHC'), ('Day', 'Fecha')]:
    for ds_label, (df, X_scaled, y, groups) in [
        ('Binary', (df_binary, X_scaled_binary, y_binary, df_binary)),
        ('Weighted', (df_weighted, X_scaled_weighted, y_weighted, df_weighted))
    ]:
        logo = LeaveOneGroupOut()
        y_true, y_pred, y_score = [], [], []
        # Ensure float32 for TensorFlow
        X_nn = X_scaled.astype(np.float32)
        y_nn = np.array(y).astype(np.float32)
        if group_field == 'NHC':
            group_vals = df['NHC']
        else:
            group_vals = df['Fecha']
        for train_idx, test_idx in logo.split(X_nn, y_nn, group_vals):
            X_train, X_test = X_nn[train_idx], X_nn[test_idx]
            y_train, y_test = y_nn[train_idx], y_nn[test_idx]
            model = create_binary_nn(X_train.shape[1])
            early_stop = EarlyStopping(patience=2, restore_best_weights=True, verbose=0)
            model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.1,
                      callbacks=[early_stop], verbose=0)
            y_score_fold = model.predict(X_test).flatten()
            y_pred_fold = (y_score_fold >= 0.5).astype(int)
            y_true.extend(y_test)
            y_pred.extend(y_pred_fold)
            y_score.extend(y_score_fold)
        key = f"{ds_label}_Bin_LOPOCV_{group_label}"
        results_nn_bin[key] = {
            'y_true': np.array(y_true),
            'y_pred': np.array(y_pred),
            'y_score': np.array(y_score),
            'metrics': collect_metrics_bin(y_true, y_pred, y_score)
        }
        print(f"\nNEURAL NETWORK | {ds_label} dataset | BINARY | LOPOCV by {group_label}")
        for metric, value in results_nn_bin[key]['metrics'].items():
            print(f"{metric}: {value:.3f}")

# ---------- MULTICLASS CLASSIFICATION ----------
results_nn_mc = {}

for group_label, group_field in [('Turn', 'NHC'), ('Day', 'Fecha')]:
    for ds_label, (df, X_scaled) in [
        ('Binary', (df_binary, X_scaled_binary)),
        ('Weighted', (df_weighted, X_scaled_weighted))
    ]:
        # Prepare multiclass data
        df_mc = df[df['Risk_Multiclass'].notna()].copy()
        drop_cols = ['NHC', 'Fecha', 'Turno', 'Aggressive', 'Self-Harm', 'Absconding', 'Risk_Multiclass', 'Risk_Binary']
        X_mc = df_mc.drop(columns=drop_cols)
        X_mc = pd.get_dummies(X_mc, drop_first=True)
        y_mc = LabelEncoder().fit_transform(df_mc['Risk_Multiclass'])
        labels_map = dict(enumerate(LabelEncoder().fit(df_mc['Risk_Multiclass']).classes_))
        # Tensorflow format
        X_mc_nn = X_mc.values.astype(np.float32)
        y_mc_nn = y_mc.astype(np.int32)
        if group_field == 'NHC':
            group_vals = df_mc['NHC']
        else:
            group_vals = df_mc['Fecha']
        logo = LeaveOneGroupOut()
        y_true_mc, y_pred_mc = [], []
        for train_idx, test_idx in logo.split(X_mc_nn, y_mc_nn, group_vals):
            X_train, X_test = X_mc_nn[train_idx], X_mc_nn[test_idx]
            y_train, y_test = y_mc_nn[train_idx], y_mc_nn[test_idx]
            model = create_multiclass_nn(X_train.shape[1], output_dim=len(labels_map))
            early_stop = EarlyStopping(patience=2, restore_best_weights=True, verbose=0)
            model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.1,
                      callbacks=[early_stop], verbose=0)
            y_score_fold = model.predict(X_test)
            y_pred_fold = np.argmax(y_score_fold, axis=1)
            y_true_mc.extend(y_test)
            y_pred_mc.extend(y_pred_fold)
        key = f"{ds_label}_MC_LOPOCV_{group_label}"
        results_nn_mc[key] = {
            'y_true': np.array(y_true_mc),
            'y_pred': np.array(y_pred_mc),
            'metrics': collect_metrics_mc(y_true_mc, y_pred_mc),
            'labels_map': labels_map
        }
        print(f"\nNEURAL NETWORK | {ds_label} dataset | MULTICLASS | LOPOCV by {group_label}")
        for metric, value in results_nn_mc[key]['metrics'].items():
            print(f"{metric}: {value:.3f}")


# In[41]:


### SHAP by shift all data entries

import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# --- FUNCTION ---
def run_shap(model, X_all, feature_names, kind='kernel', is_multiclass=False, labels_map=None):
    background = X_all[np.random.choice(X_all.shape[0], 100, replace=False)]
    sample = X_all[np.random.choice(X_all.shape[0], 10, replace=False)]
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(sample)
    if not is_multiclass:
        shap.summary_plot(shap_values, sample, feature_names=feature_names)
    else:
        for i, class_name in enumerate(labels_map.values()):
            shap.summary_plot(shap_values[i], sample, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary – Class: {class_name}")
            plt.show()

# --- BINARY FEATURES ---
print("\nSHAP – Logistic Regression (Binary, Binary Features, by TURN)")
model_logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
model_logreg.fit(X_scaled_binary, y_binary)
run_shap(model_logreg, X_scaled_binary, get_feature_names(df_binary))

print("\nSHAP – Logistic Regression (Binary, Weighted Features, by TURN)")
model_logreg_w = LogisticRegression(max_iter=1000, class_weight='balanced')
model_logreg_w.fit(X_scaled_weighted, y_weighted)
run_shap(model_logreg_w, X_scaled_weighted, get_feature_names(df_weighted))

# --- MULTICLASS FEATURES ---
print("\nSHAP – Logistic Regression (Multiclass, Binary Features, by TURN)")
df_mc = df_binary[df_binary['Risk_Multiclass'].notna()].copy()
X_mc = pd.get_dummies(df_mc.drop(columns=[
    'NHC', 'Fecha', 'Turno', 'Aggressive', 'Self-Harm', 'Absconding', 'Risk_Multiclass', 'Risk_Binary'
]))
y_mc = LabelEncoder().fit_transform(df_mc['Risk_Multiclass'])
labels_map = dict(enumerate(LabelEncoder().fit(df_mc['Risk_Multiclass']).classes_))
model_logreg_mc = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='multinomial', solver='lbfgs')
model_logreg_mc.fit(X_mc.values, y_mc)
run_shap(model_logreg_mc, X_mc.values, X_mc.columns, is_multiclass=True, labels_map=labels_map)

print("\nSHAP – Logistic Regression (Multiclass, Weighted Features, by TURN)")
df_mc_w = df_weighted[df_weighted['Risk_Multiclass'].notna()].copy()
X_mc_w = pd.get_dummies(df_mc_w.drop(columns=[
    'NHC', 'Fecha', 'Turno', 'Aggressive', 'Self-Harm', 'Absconding', 'Risk_Multiclass', 'Risk_Binary'
]))
y_mc_w = LabelEncoder().fit_transform(df_mc_w['Risk_Multiclass'])
labels_map_w = dict(enumerate(LabelEncoder().fit(df_mc_w['Risk_Multiclass']).classes_))
model_logreg_mc_w = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='multinomial', solver='lbfgs')
model_logreg_mc_w.fit(X_mc_w.values, y_mc_w)
run_shap(model_logreg_mc_w, X_mc_w.values, X_mc_w.columns, is_multiclass=True, labels_map=labels_map_w)


# In[42]:


def run_shap(model, X_all, feature_names, is_multiclass=False, labels_map=None):
    background = X_all[np.random.choice(X_all.shape[0], 100, replace=False)]
    sample = X_all[np.random.choice(X_all.shape[0], 10, replace=False)]
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(sample)
    if not is_multiclass:
        shap.summary_plot(shap_values, sample, feature_names=feature_names)
    else:
        for i, class_name in enumerate(labels_map.values()):
            shap.summary_plot(shap_values[i], sample, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary – Class: {class_name}")
            plt.show()

print("\nSHAP – Random Forest (Binary, Binary Features, by TURN)")
model_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_rf.fit(X_scaled_binary, y_binary)
run_shap(model_rf, X_scaled_binary, get_feature_names(df_binary))

print("\nSHAP – Random Forest (Binary, Weighted Features, by TURN)")
model_rf_w = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_rf_w.fit(X_scaled_weighted, y_weighted)
run_shap(model_rf_w, X_scaled_weighted, get_feature_names(df_weighted))

print("\nSHAP – Random Forest (Multiclass, Binary Features, by TURN)")
model_rf_mc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_rf_mc.fit(X_mc.values, y_mca
run_shap(model_rf_mc, X_mc.values, X_mc.columns, is_multiclass=True, labels_map=labels_map)

print("\nSHAP – Random Forest (Multiclass, Weighted Features, by TURN)")
model_rf_mc_w = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_rf_mc_w.fit(X_mc_w.values, y_mc_w)
run_shap(model_rf_mc_w, X_mc_w.values, X_mc_w.columns, is_multiclass=True, labels_map=labels_map_w)


# In[44]:


def create_binary_nn(input_dim):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

print("\nSHAP – Neural Network (Binary, Binary Features, by TURN)")
model_nn = create_binary_nn(X_scaled_binary.shape[1])
model_nn.fit(X_scaled_binary, y_binary, epochs=8, batch_size=64, validation_split=0.2, verbose=0)
background = X_scaled_binary[np.random.choice(X_scaled_binary.shape[0], 100, replace=False)]
sample = X_scaled_binary[np.random.choice(X_scaled_binary.shape[0], 10, replace=False)]
explainer = shap.KernelExplainer(model_nn.predict, background)
shap_values = explainer.shap_values(sample)
shap.summary_plot(shap_values, sample, feature_names=get_feature_names(df_binary))


# In[47]:


from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from xgboost import XGBClassifier
import numpy as np

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 4, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

best_params_bin = None
best_params_weighted = None

for ds_label, (X, y, groups) in [
    ('Binary', (X_scaled_binary, y_binary, df_binary['NHC'])),
    ('Weighted', (X_scaled_weighted, y_weighted, df_weighted['NHC']))
]:
    logo = LeaveOneGroupOut()
    model_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=(np.sum(y == 0) / np.sum(y == 1))
    )
    grid = GridSearchCV(
        estimator=model_xgb,
        param_grid=param_grid,
        cv=logo.split(X, y, groups),
        scoring='f1',
        verbose=0,    # No progress output!
        n_jobs=-1
    )
    grid.fit(X, y)
    # Final result only:
    print(f"\nBest parameters for {ds_label}: {grid.best_params_}")
    print(f"Best F1 score for {ds_label}: {grid.best_score_:.4f}")
    if ds_label == "Binary":
        best_params_bin = grid.best_params_
    else:
        best_params_weighted = grid.best_params_


# In[49]:


# For multiclass as well (change 'objective' and eval_metric):

# For multiclass (binary and weighted)
for ds_label, df in [
    ('Binary', df_binary),
    ('Weighted', df_weighted)
]:
    df_mc = df[df['Risk_Multiclass'].notna()].copy()
    drop_cols = ['NHC', 'Fecha', 'Turno', 'Aggressive', 'Self-Harm', 'Absconding', 'Risk_Multiclass', 'Risk_Binary']
    X_mc = pd.get_dummies(df_mc.drop(columns=drop_cols))
    y_mc = LabelEncoder().fit_transform(df_mc['Risk_Multiclass'])
    groups_mc = df_mc['NHC']

    print(f"\nGrid Search XGBoost ({ds_label} features, multiclass classification, LOPOCV by patient)")
    model_xgb_mc = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_mc)), eval_metric='mlogloss', random_state=42)
    grid_mc = GridSearchCV(
        estimator=model_xgb_mc,
        param_grid=param_grid,
        cv=LeaveOneGroupOut().split(X_mc.values, y_mc, groups_mc),
        scoring='f1_weighted',
        verbose=0,
        n_jobs=-1
    )
    grid_mc.fit(X_mc.values, y_mc)
    print("Best parameters:", grid_mc.best_params_)
    print("Best F1 score:", grid_mc.best_score_)
    if ds_label == "Binary":
        best_params_bin_mc = grid_mc.best_params_
    else:
        best_params_weighted_mc = grid_mc.best_params_


# In[51]:


# --- Use best_params_bin and best_params_weighted from the grid search above ---

# For Binary features
model_xgb_opt_bin = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42, 
    scale_pos_weight=(np.sum(y_binary==0)/np.sum(y_binary==1)),
    **best_params_bin
)
model_xgb_opt_bin.fit(X_scaled_binary, y_binary)
# Predictions, evaluation, etc.
y_pred_opt_bin = model_xgb_opt_bin.predict(X_scaled_binary)
y_score_opt_bin = model_xgb_opt_bin.predict_proba(X_scaled_binary)[:,1]

# For Weighted features
model_xgb_opt_weighted = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42, 
    scale_pos_weight=(np.sum(y_weighted==0)/np.sum(y_weighted==1)),
    **best_params_weighted
)
model_xgb_opt_weighted.fit(X_scaled_weighted, y_weighted)
y_pred_opt_weighted = model_xgb_opt_weighted.predict(X_scaled_weighted)
y_score_opt_weighted = model_xgb_opt_weighted.predict_proba(X_scaled_weighted)[:,1]

# Metrics
print("\n--- Optimized XGBoost (Binary features) ---")
print(classification_report(y_binary, y_pred_opt_bin))
print("AUC:", roc_auc_score(y_binary, y_score_opt_bin))
print("\n--- Optimized XGBoost (Weighted features) ---")
print(classification_report(y_weighted, y_pred_opt_weighted))
print("AUC:", roc_auc_score(y_weighted, y_score_opt_weighted))


# In[56]:


import pandas as pd

model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Net']
results_bin = [results_logreg_bin, results_rf_bin, results_xgb_bin, results_nn_bin]
group_labels = ['Turn', 'Day']
ds_labels = ['Binary', 'Weighted']

rows = []
for model, res_dict in zip(model_names, results_bin):
    for group in group_labels:
        for ds in ds_labels:
            key = f"{ds}_Bin_LOPOCV_{group}"
            if key in res_dict:
                m = res_dict[key]['metrics']
                rows.append({
                    'Model': model,
                    'Grouping': group,
                    'Dataset': ds,
                    'Accuracy': m['Accuracy'],
                    'Precision': m['Precision'],
                    'Recall': m['Recall'],
                    'F1-score': m['F1'],
                    'AUC': m['AUC'],
                })
df_results_bin = pd.DataFrame(rows)

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

styled = (
    df_results_bin.style
    .format({'Accuracy': '{:.3f}', 'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1-score': '{:.3f}', 'AUC': '{:.3f}'})
    .apply(highlight_max, subset=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC'])
    .set_properties(**{'text-align': 'center'})
    .set_caption("Comparative Model Performance (Binary Classification)")
)
styled


# In[57]:


model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Net']
results_mc = [results_logreg_mc, results_rf_mc, results_xgb_mc, results_nn_mc]
group_labels = ['Turn', 'Day']
ds_labels = ['Binary', 'Weighted']

rows_mc = []
for model, res_dict in zip(model_names, results_mc):
    for group in group_labels:
        for ds in ds_labels:
            key = f"{ds}_MC_LOPOCV_{group}"
            if key in res_dict:
                m = res_dict[key]['metrics']
                rows_mc.append({
                    'Model': model,
                    'Grouping': group,
                    'Dataset': ds,
                    'Accuracy': m['Accuracy'],
                    'Precision': m['Precision'],
                    'Recall': m['Recall'],
                    'F1-score': m['F1'],
                })
df_results_mc = pd.DataFrame(rows_mc)

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

styled_mc = (
    df_results_mc.style
    .format({'Accuracy': '{:.3f}', 'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1-score': '{:.3f}'})
    .apply(highlight_max, subset=['Accuracy', 'Precision', 'Recall', 'F1-score'])
    .set_properties(**{'text-align': 'center'})
    .set_caption("Comparative Model Performance (Multiclass Classification)")
)
styled_mc


# In[83]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network']
results_bin = [results_logreg_bin, results_rf_bin, results_xgb_bin, results_nn_bin]
key = "Binary_Bin_LOPOCV_Day"

# Construir tabla de métricas y seleccionar top 3 por F1
rows = []
for name, res_dict in zip(model_names, results_bin):
    m = res_dict[key]['metrics']
    rows.append({'Model': name, 'F1': m['F1'], 'AUC': m['AUC']})
df = pd.DataFrame(rows)
df = df.sort_values('F1', ascending=False)
top_models = df.head(3)['Model'].tolist()

plt.figure(figsize=(8, 6))
for model, res_dict in zip(model_names, results_bin):
    if model not in top_models:
        continue
    y_true = res_dict[key]['y_true']
    y_score = res_dict[key]['y_score']
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{model} (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Top 3, Binary, Day grouping)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# In[84]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network']
results_bin = [results_logreg_bin, results_rf_bin, results_xgb_bin, results_nn_bin]
key = "Weighted_Bin_LOPOCV_Turn"

# Construir tabla de métricas y seleccionar top 3 por F1
rows = []
for name, res_dict in zip(model_names, results_bin):
    m = res_dict[key]['metrics']
    rows.append({'Model': name, 'F1': m['F1'], 'AUC': m['AUC']})
df = pd.DataFrame(rows)
df = df.sort_values('F1', ascending=False)
top_models = df.head(3)['Model'].tolist()

plt.figure(figsize=(8, 6))
for model, res_dict in zip(model_names, results_bin):
    if model not in top_models:
        continue
    y_true = res_dict[key]['y_true']
    y_score = res_dict[key]['y_score']
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{model} (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Top 3, Weighted, Turn grouping)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# In[74]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Configuración: mejores modelos y resultados
mejores_bin = [
    ('Random Forest', results_rf_bin, 'Turn', 'Binary'),
    ('Random Forest', results_rf_bin, 'Turn', 'Weighted'),
    ('XGBoost', results_xgb_bin, 'Day', 'Binary'),
    ('XGBoost', results_xgb_bin, 'Day', 'Weighted'),
]

for nombre, results_dict, group, ds in mejores_bin:
    key = f"{ds}_Bin_LOPOCV_{group}"
    y_true = results_dict[key]['y_true']
    y_pred = results_dict[key]['y_pred']
    # Puedes poner tus propias etiquetas aquí si lo necesitas:
    labels = ['No Risk', 'Risk']  # O usa tu codificación real
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    im = disp.plot(ax=ax, cmap="Blues", values_format='d', colorbar=False)
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    plt.title(f"Confusion Matrix\n{nombre} ({group}, {ds})")
    plt.tight_layout()
    plt.show()


# In[62]:


get_ipython().system('pip install statsmodels')


# In[68]:


import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
import itertools

model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Net']
results_bin = [results_logreg_bin, results_rf_bin, results_xgb_bin, results_nn_bin]
scenarios = [
    ('Turn', 'Binary'),
    ('Turn', 'Weighted'),
    ('Day', 'Binary'),
    ('Day', 'Weighted')
]

summary_rows = []

for group, ds in scenarios:
    key = f"{ds}_Bin_LOPOCV_{group}"
    for (i, model1), (j, model2) in itertools.combinations(enumerate(model_names), 2):
        y_true = results_bin[i][key]['y_true']
        y_pred1 = results_bin[i][key]['y_pred']
        y_pred2 = results_bin[j][key]['y_pred']
        # McNemar contingency table
        table = np.zeros((2,2))
        for a in range(len(y_true)):
            correct1 = y_pred1[a] == y_true[a]
            correct2 = y_pred2[a] == y_true[a]
            table[int(not correct1), int(not correct2)] += 1
        result = mcnemar(table, exact=True)
        summary_rows.append({
            'Scenario': f"{group}/{ds}",
            'Model 1': model1,
            'Model 2': model2,
            'McNemar p-value': result.pvalue,
            'Significant (p < 0.05)': 'Yes' if result.pvalue < 0.05 else 'No'
        })

df_mcnemar = pd.DataFrame(summary_rows)

# Visualización: resalta los p-valores significativos
def highlight_sig(val):
    color = 'background-color: lightcoral; font-weight: bold;' if isinstance(val, float) and val < 0.05 else ''
    return color

styled = (
    df_mcnemar.style
    .format({'McNemar p-value': '{:.4f}'})
    .applymap(highlight_sig, subset=['McNemar p-value'])
    .set_properties(**{'text-align': 'center'})
    .set_caption("Pairwise McNemar Test (Binary Classification Scenarios)")
)

styled


# In[72]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Configura los escenarios y resultados multiclass
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Net']
results_mc = [results_logreg_mc, results_rf_mc, results_xgb_mc, results_nn_mc]
scenarios = [
    ('Turn', 'Binary'),
    ('Turn', 'Weighted'),
    ('Day', 'Binary'),
    ('Day', 'Weighted')
]

# Por cada escenario, elige el modelo con mayor Accuracy
for group, ds in scenarios:
    key = f"{ds}_MC_LOPOCV_{group}"
    # Encuentra el mejor modelo por Accuracy
    accuracies = [results_mc[i][key]['metrics']['Accuracy'] for i in range(len(model_names))]
    idx_best = np.argmax(accuracies)
    best_model = results_mc[idx_best]
    best_model_name = model_names[idx_best]
    labels_map = best_model[key]['labels_map']
    y_true_mc = best_model[key]['y_true']
    y_pred_mc = best_model[key]['y_pred']
    # Matriz de confusión absoluta
    cm = confusion_matrix(y_true_mc, y_pred_mc)
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_map.values())
    im = disp.plot(ax=ax, cmap="Blues", values_format='d', colorbar=False)
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    plt.title(f"Confusion Matrix\n{best_model_name} ({group}, {ds})")
    plt.tight_layout()
    plt.show()


# In[76]:


from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from scipy.stats import wilcoxon
import numpy as np
import pandas as pd

# Modelos y resultados multiclass
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Net']
results_mc = [results_logreg_mc, results_rf_mc, results_xgb_mc, results_nn_mc]
scenarios = [
    ('Turn', 'Binary', df_binary),
    ('Turn', 'Weighted', df_weighted),
    ('Day', 'Binary', df_binary),
    ('Day', 'Weighted', df_weighted)
]

summary_rows = []

def accuracies_by_group(y_true_all, y_pred_all, groups):
    logo = LeaveOneGroupOut()
    accs = []
    X_dummy = np.zeros((len(y_true_all), 1))
    for train_idx, test_idx in logo.split(X_dummy, y_true_all, groups):
        y_true = np.array(y_true_all)[test_idx]
        y_pred = np.array(y_pred_all)[test_idx]
        accs.append(accuracy_score(y_true, y_pred))
    return accs

for group, ds, df in scenarios:
    key = f"{ds}_MC_LOPOCV_{group}"
    # Escoge los dos mejores modelos por Accuracy global
    accs_global = [results_mc[i][key]['metrics']['Accuracy'] for i in range(len(model_names))]
    idx_best = np.argsort(accs_global)[-2:][::-1]
    i, j = idx_best[0], idx_best[1]
    model1, model2 = model_names[i], model_names[j]
    res1 = results_mc[i][key]
    res2 = results_mc[j][key]
    y_true1, y_pred1 = res1['y_true'], res1['y_pred']
    y_true2, y_pred2 = res2['y_true'], res2['y_pred']
    # Consigue grupos SOLO de las filas multiclass
    mask = df['Risk_Multiclass'].notna()
    if group == 'Turn':
        groups_used = df.loc[mask, 'NHC']
    else:
        groups_used = df.loc[mask, 'Fecha']
    # Accuracies por grupo
    accs1 = accuracies_by_group(y_true1, y_pred1, groups_used)
    accs2 = accuracies_by_group(y_true2, y_pred2, groups_used)
    # Wilcoxon test
    stat, p = wilcoxon(accs1, accs2)
    summary_rows.append({
        'Scenario': f"{group}/{ds}",
        'Best Model': model1,
        'Second Best Model': model2,
        'Accuracy 1': accs_global[i],
        'Accuracy 2': accs_global[j],
        'Wilcoxon p-value': p,
        'Significant (p < 0.05)': 'Yes' if p < 0.05 else 'No'
    })

df_wilcoxon = pd.DataFrame(summary_rows)

# Visualización: resalta los p-valores significativos
def highlight_sig(val):
    color = 'background-color: lightcoral; font-weight: bold;' if isinstance(val, float) and val < 0.05 else ''
    return color

styled_wilcoxon = (
    df_wilcoxon.style
    .format({'Accuracy 1': '{:.3f}', 'Accuracy 2': '{:.3f}', 'Wilcoxon p-value': '{:.4f}'})
    .applymap(highlight_sig, subset=['Wilcoxon p-value'])
    .set_properties(**{'text-align': 'center'})
    .set_caption("Wilcoxon Test Between Best Multiclass Models per Scenario")
)

styled_wilcoxon


# In[77]:


from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from scipy.stats import wilcoxon
import numpy as np
import pandas as pd
import itertools

# Modelos y resultados multiclass
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Net']
results_mc = [results_logreg_mc, results_rf_mc, results_xgb_mc, results_nn_mc]
scenarios = [
    ('Turn', 'Binary', df_binary),
    ('Turn', 'Weighted', df_weighted),
    ('Day', 'Binary', df_binary),
    ('Day', 'Weighted', df_weighted)
]

def accuracies_by_group(y_true_all, y_pred_all, groups):
    logo = LeaveOneGroupOut()
    accs = []
    X_dummy = np.zeros((len(y_true_all), 1))
    for train_idx, test_idx in logo.split(X_dummy, y_true_all, groups):
        y_true = np.array(y_true_all)[test_idx]
        y_pred = np.array(y_pred_all)[test_idx]
        accs.append(accuracy_score(y_true, y_pred))
    return accs

summary_rows = []

for group, ds, df in scenarios:
    key = f"{ds}_MC_LOPOCV_{group}"
    # Consigue grupos SOLO de las filas multiclass
    mask = df['Risk_Multiclass'].notna()
    if group == 'Turn':
        groups_used = df.loc[mask, 'NHC']
    else:
        groups_used = df.loc[mask, 'Fecha']
    # Para todas las combinaciones de pares de modelos
    for (i, name1), (j, name2) in itertools.combinations(enumerate(model_names), 2):
        res1 = results_mc[i][key]
        res2 = results_mc[j][key]
        y_true1, y_pred1 = res1['y_true'], res1['y_pred']
        y_true2, y_pred2 = res2['y_true'], res2['y_pred']
        # Accuracies por grupo
        accs1 = accuracies_by_group(y_true1, y_pred1, groups_used)
        accs2 = accuracies_by_group(y_true2, y_pred2, groups_used)
        # Wilcoxon test
        stat, p = wilcoxon(accs1, accs2)
        summary_rows.append({
            'Scenario': f"{group}/{ds}",
            'Model 1': name1,
            'Model 2': name2,
            'Wilcoxon p-value': p,
            'Significant (p < 0.05)': 'Yes' if p < 0.05 else 'No'
        })

df_wilcoxon = pd.DataFrame(summary_rows)

# Visualización: resalta los p-valores significativos
def highlight_sig(val):
    color = 'background-color: lightcoral; font-weight: bold;' if isinstance(val, float) and val < 0.05 else ''
    return color

styled_wilcoxon = (
    df_wilcoxon.style
    .format({'Wilcoxon p-value': '{:.4f}'})
    .applymap(highlight_sig, subset=['Wilcoxon p-value'])
    .set_properties(**{'text-align': 'center'})
    .set_caption("Wilcoxon Test: All Pairwise Comparisons, Multiclass, All Scenarios")
)

styled_wilcoxon


# In[78]:


from sklearn.metrics import classification_report

# Ejemplo para un mejor modelo
print(classification_report(y_true_mc, y_pred_mc, target_names=list(labels_map.values())))


# In[98]:


import pandas as pd
import matplotlib.pyplot as plt

# ---- TUS DATOS: (se asume que results_*_bin y results_*_mc ya existen en memoria) ----

model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Net']
group_labels = ['Turn', 'Day']
ds_labels = ['Binary', 'Weighted']

# ----- BINARIO -----
results_bin = [results_logreg_bin, results_rf_bin, results_xgb_bin, results_nn_bin]
rows = []
for model, res_dict in zip(model_names, results_bin):
    for group in group_labels:
        for ds in ds_labels:
            key = f"{ds}_Bin_LOPOCV_{group}"
            if key in res_dict:
                m = res_dict[key]['metrics']
                rows.append({
                    'Model': model,
                    'Grouping': group,
                    'Dataset': ds,
                    'Accuracy': m['Accuracy'],
                    'Precision': m['Precision'],
                    'Recall': m['Recall'],
                    'F1-score': m['F1'],
                    'AUC': m['AUC'],
                })
df_results_bin = pd.DataFrame(rows)

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

styled_bin = (
    df_results_bin.style
    .format({'Accuracy': '{:.3f}', 'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1-score': '{:.3f}', 'AUC': '{:.3f}'})
    .apply(highlight_max, subset=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC'])
    .set_properties(**{'text-align': 'center'})
    .set_caption("Comparative Model Performance (Binary Classification)")
)

# ----- MULTICLASE -----
results_mc = [results_logreg_mc, results_rf_mc, results_xgb_mc, results_nn_mc]
rows_mc = []
for model, res_dict in zip(model_names, results_mc):
    for group in group_labels:
        for ds in ds_labels:
            key = f"{ds}_MC_LOPOCV_{group}"
            if key in res_dict:
                m = res_dict[key]['metrics']
                rows_mc.append({
                    'Model': model,
                    'Grouping': group,
                    'Dataset': ds,
                    'Accuracy': m['Accuracy'],
                    'Precision': m['Precision'],
                    'Recall': m['Recall'],
                    'F1-score': m['F1'],
                })
df_results_mc = pd.DataFrame(rows_mc)

styled_mc = (
    df_results_mc.style
    .format({'Accuracy': '{:.3f}', 'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1-score': '{:.3f}'})
    .apply(highlight_max, subset=['Accuracy', 'Precision', 'Recall', 'F1-score'])
    .set_properties(**{'text-align': 'center'})
    .set_caption("Comparative Model Performance (Multiclass Classification)")
)

# ---- Mostrar las tablas en Jupyter Notebook ----
display(styled_bin)
display(styled_mc)

# ---- Guardar las tablas en Excel y CSV para anexos ----
df_results_bin.to_excel("tabla_metricas_binarias.xlsx", index=False)
df_results_bin.to_csv("tabla_metricas_binarias.csv", index=False)
df_results_mc.to_excel("tabla_metricas_multiclase.xlsx", index=False)
df_results_mc.to_csv("tabla_metricas_multiclase.csv", index=False)

# ---- Visualización rápida de comparación por métrica y modelo ----
def plot_metric_bars(df, title, metrics=['Accuracy', 'Precision', 'Recall', 'F1-score'], hue='Model'):
    for metric in metrics:
        plt.figure(figsize=(10,5))
        for group in df['Grouping'].unique():
            df_group = df[df['Grouping'] == group]
            for ds in df['Dataset'].unique():
                df_ds = df_group[df_group['Dataset'] == ds]
                plt.bar([f"{m}\n({ds}-{group})" for m in df_ds['Model']], df_ds[metric], label=f"{ds}-{group}")
        plt.ylabel(metric)
        plt.title(f"{title}: {metric}")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

plot_metric_bars(df_results_bin, "Binary Classification")
plot_metric_bars(df_results_mc, "Multiclass Classification", metrics=['Accuracy', 'Precision', 'Recall', 'F1-score'])

print("¡Tablas y visualizaciones generadas y guardadas!")

# ---- (Opcional) Análisis adicional: exportar resumen global ----
summary = {
    "Best Accuracy (Binary)": df_results_bin.loc[df_results_bin['Accuracy'].idxmax()],
    "Best F1-score (Binary)": df_results_bin.loc[df_results_bin['F1-score'].idxmax()],
    "Best Accuracy (Multiclass)": df_results_mc.loc[df_results_mc['Accuracy'].idxmax()],
    "Best F1-score (Multiclass)": df_results_mc.loc[df_results_mc['F1-score'].idxmax()],
}
for k, v in summary.items():
    print(f"\n{k}:\n{v}")

# Listo para anexar a tu TFM


# In[100]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define tus etiquetas de clase reales, por ejemplo para binario: class_labels = [0,1]
class_labels = [0, 1]  # Cambia según tus datos

# Estructura esperada:
# y_true_dict = {('Binary', 'Turn'): y_true_bin_turn, ...}
# y_pred_dict = {('Logistic Regression', 'Binary', 'Turn'): y_pred_logreg_bin_turn, ...}

# ----------- EJEMPLO DE SIMULACIÓN: Reemplaza por tus arrays reales -----------
# Este bloque simula datos. BORRA y usa tus arrays.
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Net']
group_labels = ['Turn', 'Day']
ds_labels = ['Binary', 'Weighted']

y_true_dict = {}
y_pred_dict = {}
for ds in ds_labels:
    for group in group_labels:
        key = (ds, group)
        y_true_dict[key] = np.random.randint(0, len(class_labels), 100)
        for model in model_names:
            y_pred_dict[(model, ds, group)] = np.random.randint(0, len(class_labels), 100)
# ------------------------------------------------------------------------------

def save_conf_matrix(y_true, y_pred, labels, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

for ds in ds_labels:
    for group in group_labels:
        key = (ds, group)
        y_true = y_true_dict[key]
        for model in model_names:
            y_pred = y_pred_dict[(model, ds, group)]
            title = f"{model} - {ds} - {group}"
            fname = f"conf_matrix_{model.replace(' ','_')}_{ds}_{group}.png"
            save_conf_matrix(y_true, y_pred, labels=class_labels, title=title, fname=fname)

print("Matrices de confusión generadas y guardadas para todos los modelos y configuraciones.")

