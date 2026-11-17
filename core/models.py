"""Modul pelatihan model untuk level Easy, Medium, dan Hard (Imbalance, Tuning, Leaderboard, Ensembling)."""

import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from core.preprocessing import build_column_transformer


def train_easy_model(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    impute_strategy: str = "median",
    test_size: float = 0.2,
    c_param: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Melatih model dasar (Linear Regression / Logistic Regression) pada modul Easy."""
    df_clean = df.dropna(subset=[target_col]).copy()
    feature_cols = [
        c for c in df_clean.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df_clean[c])
    ]

    if impute_strategy == "drop":
        df_clean = df_clean.dropna(subset=feature_cols).reset_index(drop=True)
        imp_step = SimpleImputer(strategy="median")
    else:
        imp_step = SimpleImputer(strategy=impute_strategy)

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    stratify_y = y if (task_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify_y
    )

    if task_type == "classification":
        estimator = LogisticRegression(C=c_param, max_iter=500, random_state=seed)
        model_name = "Logistic Regression"
    else:
        estimator = LinearRegression()
        model_name = "Linear Regression"

    pipe = Pipeline(
        steps=[
            ("imputer", imp_step),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    if task_type == "classification":
        avg_mode = "binary" if y.nunique() == 2 else "macro"
        metrics = {
            "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "Precision": round(float(precision_score(y_test, y_pred, average=avg_mode, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, y_pred, average=avg_mode, zero_division=0)), 4),
            "F1_Score": round(float(f1_score(y_test, y_pred, average=avg_mode, zero_division=0)), 4),
        }
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics = {
            "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "RMSE": round(rmse, 4),
            "R2_Score": round(float(r2_score(y_test, y_pred)), 4),
        }

    coefs = getattr(pipe.named_steps["model"], "coef_", np.zeros(len(feature_cols)))
    if coefs.ndim > 1:
        coefs = np.mean(np.abs(coefs), axis=0)

    equivalent_code = f'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import {"LogisticRegression" if task_type == "classification" else "LinearRegression"}

GLOBAL_SEED = {seed}
X = df[{feature_cols}]
y = df["{target_col}"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state=GLOBAL_SEED
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="{impute_strategy if impute_strategy != 'drop' else 'median'}")),
    ("scaler", StandardScaler()),
    ("model", {"LogisticRegression(C=" + str(c_param) + ", max_iter=500, random_state=GLOBAL_SEED)" if task_type == "classification" else "LinearRegression()"})
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
'''

    return {
        "model_name": model_name,
        "pipeline": pipe,
        "metrics": metrics,
        "y_test": np.asarray(y_test),
        "y_pred": np.asarray(y_pred),
        "feature_names": feature_cols,
        "importances": np.asarray(coefs),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "code": equivalent_code,
    }


def train_medium_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    model_choice: str,
    num_imputer_type: str,
    scaler_type: str,
    encoder_type: str,
    cv_folds: int,
    test_size: float,
    hyperparams: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Melatih Pipeline + ColumnTransformer pada modul Medium dengan evaluasi K-Fold CV.
    Pilihan model_choice: 'KNN', 'Decision Tree', 'Random Forest', 'SVM'.
    """
    df_clean = df.dropna(subset=[target_col]).copy()
    feature_cols = [c for c in df_clean.columns if c != target_col]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_clean[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df_clean[c])]

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    stratify_y = y if (task_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= cv_folds) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify_y
    )

    preprocessor = build_column_transformer(
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_imputer_type=num_imputer_type,
        scaler_type=scaler_type,
        encoder_type=encoder_type,
        task_type=task_type,
        seed=seed,
    )

    if task_type == "classification":
        if model_choice == "KNN":
            estimator = KNeighborsClassifier(n_neighbors=int(hyperparams.get("n_neighbors", 5)))
        elif model_choice == "Decision Tree":
            estimator = DecisionTreeClassifier(
                max_depth=int(hyperparams.get("max_depth", 6)), random_state=seed
            )
        elif model_choice == "SVM":
            estimator = SVC(C=float(hyperparams.get("C", 1.0)), probability=True, random_state=seed)
        else:
            estimator = RandomForestClassifier(
                n_estimators=int(hyperparams.get("n_estimators", 100)),
                max_depth=int(hyperparams.get("max_depth", 8)),
                random_state=seed,
                n_jobs=-1,
            )
    else:
        if model_choice == "KNN":
            estimator = KNeighborsRegressor(n_neighbors=int(hyperparams.get("n_neighbors", 5)))
        elif model_choice == "Decision Tree":
            estimator = DecisionTreeRegressor(
                max_depth=int(hyperparams.get("max_depth", 6)), random_state=seed
            )
        elif model_choice == "SVM":
            estimator = SVR(C=float(hyperparams.get("C", 1.0)))
        else:
            estimator = RandomForestRegressor(
                n_estimators=int(hyperparams.get("n_estimators", 100)),
                max_depth=int(hyperparams.get("max_depth", 8)),
                random_state=seed,
                n_jobs=-1,
            )

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    if task_type == "classification" and stratify_y is not None:
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scoring_metric = "f1_macro" if y.nunique() > 2 else "f1"
    elif task_type == "classification":
        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scoring_metric = "accuracy"
    else:
        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scoring_metric = "r2"

    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv_splitter, scoring=scoring_metric)

    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)

    if task_type == "classification":
        avg_mode = "binary" if y.nunique() == 2 else "macro"
        metrics = {
            f"CV_{scoring_metric.upper()}_Mean": round(float(np.mean(cv_scores)), 4),
            f"CV_{scoring_metric.upper()}_Std": round(float(np.std(cv_scores)), 4),
            "Test_Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "Test_Precision": round(float(precision_score(y_test, y_pred, average=avg_mode, zero_division=0)), 4),
            "Test_Recall": round(float(recall_score(y_test, y_pred, average=avg_mode, zero_division=0)), 4),
            "Test_F1": round(float(f1_score(y_test, y_pred, average=avg_mode, zero_division=0)), 4),
        }
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics = {
            "CV_R2_Mean": round(float(np.mean(cv_scores)), 4),
            "CV_R2_Std": round(float(np.std(cv_scores)), 4),
            "Test_MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "Test_RMSE": round(rmse, 4),
            "Test_R2": round(float(r2_score(y_test, y_pred)), 4),
        }

    try:
        transformed_names: List[str] = list(full_pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        transformed_names = num_cols + cat_cols

    fitted_model = full_pipeline.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        importances = np.asarray(fitted_model.feature_importances_)
    else:
        X_trans = full_pipeline.named_steps["preprocessor"].transform(X_test)
        importances = np.array(
            [
                abs(np.corrcoef(X_trans[:, i], y_pred)[0, 1]) if np.std(X_trans[:, i]) > 0 else 0.0
                for i in range(X_trans.shape[1])
            ]
        )
        importances = np.nan_to_num(importances, 0.0)

    equivalent_code = f'''import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder

GLOBAL_SEED = {seed}
num_cols = {num_cols}
cat_cols = {cat_cols}

X = df[num_cols + cat_cols]
y = df["{target_col}"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state=GLOBAL_SEED
)

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", {"KNNImputer(n_neighbors=5)" if num_imputer_type == "knn" else f"SimpleImputer(strategy='{num_imputer_type}')"}),
        ("scaler", {"MinMaxScaler()" if scaler_type == "minmax" else ("RobustScaler()" if scaler_type == "robust" else "StandardScaler()")})
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", {"OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)" if encoder_type == "ordinal" else ("TargetEncoder(random_state=GLOBAL_SEED)" if encoder_type == "target" else "OneHotEncoder(handle_unknown='ignore', sparse_output=False)")})
    ]), cat_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", {estimator.__class__.__name__}())
])

cv = KFold(n_splits={cv_folds}, shuffle=True, random_state=GLOBAL_SEED)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="{scoring_metric}")
print("CV Mean:", cv_scores.mean(), "CV Std:", cv_scores.std())

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
'''

    return {
        "model_name": model_choice,
        "pipeline": full_pipeline,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "y_test": np.asarray(y_test),
        "y_pred": np.asarray(y_pred),
        "feature_names": transformed_names,
        "importances": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "code": equivalent_code,
    }


def _get_xgboost_estimator(task_type: str, seed: int, scale_pos_weight: float = 1.0, **kwargs: Any) -> Any:
    """Lazy-load XGBoost agar halaman lain tidak terbebani waktu muat."""
    try:
        import xgboost as xgb
        if task_type == "classification":
            return xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                subsample=kwargs.get("subsample", 0.8),
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
                eval_metric="logloss",
                n_jobs=-1,
            )
        return xgb.XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.1),
            subsample=kwargs.get("subsample", 0.8),
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        if task_type == "classification":
            return GradientBoostingClassifier(random_state=seed)
        return GradientBoostingRegressor(random_state=seed)


def run_hard_optimization(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    imbalance_method: str = "smote",
    feature_selection_method: str = "rfe",
    n_selected_features: int = 8,
    tuning_method: str = "optuna",
    n_trials: int = 12,
    cv_folds: int = 4,
    ensemble_type: str = "voting",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Menjalankan rangkaian lengkap Modul Hard:
    1. Split & Imputasi + Scaling
    2. Penanganan Imbalanced Data (None / Class Weight / SMOTE)
    3. Seleksi Fitur (All / Feature Importance / RFE)
    4. Leaderboard banyak model + Ensembling (Voting & Stacking)
    5. Hyperparameter Tuning (GridSearch / RandomizedSearch / Optuna) pada XGBoost/GradientBoosting
    6. Perhitungan ROC-AUC, PR Curve, dan Learning Curve
    """
    df_clean = df.dropna(subset=[target_col]).copy()
    num_cols = [
        c for c in df_clean.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df_clean[c])
    ]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_raw = pd.DataFrame(
        scaler.fit_transform(imputer.fit_transform(df_clean[num_cols])),
        columns=num_cols,
    )
    y_raw = df_clean[target_col].values

    is_binary_clf = task_type == "classification" and len(np.unique(y_raw)) == 2
    stratify_y = y_raw if (task_type == "classification" and len(np.unique(y_raw)) > 1) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=seed, stratify=stratify_y
    )

    # 1. Seleksi Fitur (Feature Selection)
    n_keep = max(2, min(len(num_cols), int(n_selected_features)))
    if feature_selection_method == "rfe" and n_keep < len(num_cols):
        base_sel = (
            RandomForestClassifier(n_estimators=50, max_depth=6, random_state=seed, n_jobs=-1)
            if task_type == "classification"
            else RandomForestRegressor(n_estimators=50, max_depth=6, random_state=seed, n_jobs=-1)
        )
        rfe = RFE(estimator=base_sel, n_features_to_select=n_keep, step=1)
        rfe.fit(X_train, y_train)
        selected_cols = [col for col, keep in zip(num_cols, rfe.support_) if keep]
    elif feature_selection_method == "importance" and n_keep < len(num_cols):
        base_sel = (
            RandomForestClassifier(n_estimators=60, max_depth=6, random_state=seed, n_jobs=-1)
            if task_type == "classification"
            else RandomForestRegressor(n_estimators=60, max_depth=6, random_state=seed, n_jobs=-1)
        )
        base_sel.fit(X_train, y_train)
        top_idx = np.argsort(base_sel.feature_importances_)[::-1][:n_keep]
        selected_cols = [num_cols[i] for i in top_idx]
    else:
        selected_cols = list(num_cols)

    X_train_sel = X_train[selected_cols].copy()
    X_test_sel = X_test[selected_cols].copy()

    # 2. Penanganan Imbalanced Data (Khusus Klasifikasi)
    class_counts_before = pd.Series(y_train).value_counts().to_dict()
    use_class_weight = imbalance_method == "class_weight" and task_type == "classification"
    scale_pos = 1.0

    if task_type == "classification" and is_binary_clf:
        neg_cnt = max(1, int(np.sum(y_train == 0)))
        pos_cnt = max(1, int(np.sum(y_train == 1)))
        if use_class_weight:
            scale_pos = float(neg_cnt / pos_cnt)

    X_train_res, y_train_res = X_train_sel.copy(), y_train.copy()
    if task_type == "classification" and imbalance_method == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            min_class_count = int(pd.Series(y_train).value_counts().min())
            if min_class_count > 2:
                k_neighbors = min(5, min_class_count - 1)
                smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
                X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)
        except Exception:
            pass

    class_counts_after = pd.Series(y_train_res).value_counts().to_dict()

    # 3. Siapkan Daftar Model Kandidat
    cw_arg = "balanced" if use_class_weight else None
    candidates: List[Tuple[str, str, Any]] = []

    if task_type == "classification":
        m_lr = LogisticRegression(class_weight=cw_arg, max_iter=500, random_state=seed)
        m_rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight=cw_arg, random_state=seed, n_jobs=-1)
        m_et = ExtraTreesClassifier(n_estimators=100, max_depth=8, class_weight=cw_arg, random_state=seed, n_jobs=-1)
        m_gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
        m_xgb = _get_xgboost_estimator("classification", seed=seed, scale_pos_weight=scale_pos)

        candidates = [
            ("Logistic Regression", "Baseline", m_lr),
            ("Random Forest", "Tree Bagging", m_rf),
            ("Extra Trees", "Tree Bagging", m_et),
            ("Gradient Boosting", "Boosting", m_gb),
            ("XGBoost (Default)", "Boosting", m_xgb),
        ]
    else:
        m_lr = Ridge(alpha=1.0, random_state=seed)
        m_rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
        m_et = ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
        m_gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=seed)
        m_xgb = _get_xgboost_estimator("regression", seed=seed)

        candidates = [
            ("Ridge Regression", "Baseline", m_lr),
            ("Random Forest", "Tree Bagging", m_rf),
            ("Extra Trees", "Tree Bagging", m_et),
            ("Gradient Boosting", "Boosting", m_gb),
            ("XGBoost (Default)", "Boosting", m_xgb),
        ]

    # 4. Hyperparameter Tuning (GridSearch / RandomizedSearch / Optuna) pada XGBoost
    n_trials_safe = max(4, min(30, int(n_trials)))
    cv_splitter = (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        if task_type == "classification"
        else KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    )
    scoring_name = "f1" if is_binary_clf else ("f1_macro" if task_type == "classification" else "r2")

    best_params: Dict[str, Any] = {}
    tuning_history: List[Dict[str, Any]] = []

    if tuning_method == "optuna":
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial: Any) -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                }
                est = _get_xgboost_estimator(
                    task_type, seed=seed, scale_pos_weight=scale_pos, **params
                )
                scores = cross_val_score(est, X_train_res, y_train_res, cv=3, scoring=scoring_name)
                mean_s = float(np.mean(scores))
                tuning_history.append({"Trial": trial.number + 1, **params, "CV_Score": round(mean_s, 4)})
                return mean_s

            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=n_trials_safe, show_progress_bar=False)
            best_params = study.best_params
        except Exception:
            best_params = {"n_estimators": 125, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.85}
    elif tuning_method == "grid":
        param_grid = {
            "n_estimators": [75, 150],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.15],
        }
        base_xgb = _get_xgboost_estimator(task_type, seed=seed, scale_pos_weight=scale_pos)
        search = GridSearchCV(base_xgb, param_grid=param_grid, cv=3, scoring=scoring_name, n_jobs=-1)
        search.fit(X_train_res, y_train_res)
        best_params = search.best_params_
        for idx, (p, s) in enumerate(zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])):
            tuning_history.append({"Trial": idx + 1, **p, "CV_Score": round(float(s), 4)})
    else:
        param_dist = {
            "n_estimators": [50, 75, 100, 125, 150, 175, 200],
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
        }
        base_xgb = _get_xgboost_estimator(task_type, seed=seed, scale_pos_weight=scale_pos)
        search = RandomizedSearchCV(
            base_xgb,
            param_distributions=param_dist,
            n_iter=min(n_trials_safe, 15),
            cv=3,
            scoring=scoring_name,
            random_state=seed,
            n_jobs=-1,
        )
        search.fit(X_train_res, y_train_res)
        best_params = search.best_params_
        for idx, (p, s) in enumerate(zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])):
            tuning_history.append({"Trial": idx + 1, **p, "CV_Score": round(float(s), 4)})

    tuned_xgb = _get_xgboost_estimator(
        task_type, seed=seed, scale_pos_weight=scale_pos, **best_params
    )
    candidates.append((f"XGBoost (Tuned - {tuning_method.upper()})", "Tuned Boosting", tuned_xgb))

    # 5. Ensembling (Voting & Stacking)
    if task_type == "classification":
        if ensemble_type == "stacking":
            ens_model = StackingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=80, max_depth=7, random_state=seed, n_jobs=-1)),
                    ("xgb", _get_xgboost_estimator("classification", seed=seed, scale_pos_weight=scale_pos, **best_params)),
                ],
                final_estimator=LogisticRegression(),
                cv=3,
            )
            ens_label = "Stacking Ensemble (RF + Tuned XGB)"
        else:
            ens_model = VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=80, max_depth=7, random_state=seed, n_jobs=-1)),
                    ("et", ExtraTreesClassifier(n_estimators=80, max_depth=7, random_state=seed, n_jobs=-1)),
                    ("xgb", _get_xgboost_estimator("classification", seed=seed, scale_pos_weight=scale_pos, **best_params)),
                ],
                voting="soft",
            )
            ens_label = "Soft Voting Ensemble (RF + ET + Tuned XGB)"
    else:
        if ensemble_type == "stacking":
            ens_model = StackingRegressor(
                estimators=[
                    ("rf", RandomForestRegressor(n_estimators=80, max_depth=7, random_state=seed, n_jobs=-1)),
                    ("xgb", _get_xgboost_estimator("regression", seed=seed, **best_params)),
                ],
                final_estimator=Ridge(),
                cv=3,
            )
            ens_label = "Stacking Ensemble (RF + Tuned XGB)"
        else:
            ens_model = VotingRegressor(
                estimators=[
                    ("rf", RandomForestRegressor(n_estimators=80, max_depth=7, random_state=seed, n_jobs=-1)),
                    ("et", ExtraTreesRegressor(n_estimators=80, max_depth=7, random_state=seed, n_jobs=-1)),
                    ("xgb", _get_xgboost_estimator("regression", seed=seed, **best_params)),
                ]
            )
            ens_label = "Voting Ensemble (RF + ET + Tuned XGB)"

    candidates.append((ens_label, "Ensemble", ens_model))

    # 6. Latih & Evaluasi Seluruh Kandidat untuk Tabel Leaderboard
    leaderboard_rows: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}
    probs_map: Dict[str, np.ndarray] = {}
    preds_map: Dict[str, np.ndarray] = {}

    for m_name, m_cat, est in candidates:
        t0 = time.perf_counter()
        cv_s = cross_val_score(est, X_train_res, y_train_res, cv=cv_splitter, scoring=scoring_name)
        est.fit(X_train_res, y_train_res)
        elapsed = round(time.perf_counter() - t0, 2)

        y_p = est.predict(X_test_sel)
        preds_map[m_name] = y_p
        trained_models[m_name] = est

        if task_type == "classification":
            avg_m = "binary" if is_binary_clf else "macro"
            f1_v = round(float(f1_score(y_test, y_p, average=avg_m, zero_division=0)), 4)
            acc_v = round(float(accuracy_score(y_test, y_p)), 4)
            prec_v = round(float(precision_score(y_test, y_p, average=avg_m, zero_division=0)), 4)
            rec_v = round(float(recall_score(y_test, y_p, average=avg_m, zero_division=0)), 4)

            roc_v = 0.0
            if is_binary_clf and hasattr(est, "predict_proba"):
                prob_pos = est.predict_proba(X_test_sel)[:, 1]
                probs_map[m_name] = prob_pos
                roc_v = round(float(roc_auc_score(y_test, prob_pos)), 4)

            leaderboard_rows.append(
                {
                    "Model": m_name,
                    "Kategori": m_cat,
                    "CV_Mean": round(float(np.mean(cv_s)), 4),
                    "Test_F1": f1_v,
                    "Test_ROC_AUC": roc_v,
                    "Test_Precision": prec_v,
                    "Test_Recall": rec_v,
                    "Test_Accuracy": acc_v,
                    "Waktu_Detik": elapsed,
                }
            )
        else:
            rmse_v = round(float(np.sqrt(mean_squared_error(y_test, y_p))), 4)
            mae_v = round(float(mean_absolute_error(y_test, y_p)), 4)
            r2_v = round(float(r2_score(y_test, y_p)), 4)
            leaderboard_rows.append(
                {
                    "Model": m_name,
                    "Kategori": m_cat,
                    "CV_Mean": round(float(np.mean(cv_s)), 4),
                    "Test_R2": r2_v,
                    "Test_RMSE": rmse_v,
                    "Test_MAE": mae_v,
                    "Waktu_Detik": elapsed,
                }
            )

    sort_col = "Test_F1" if task_type == "classification" else "Test_R2"
    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    leaderboard_df.insert(0, "Peringkat", range(1, len(leaderboard_df) + 1))

    best_model_name = str(leaderboard_df.iloc[0]["Model"])
    best_model_obj = trained_models[best_model_name]
    best_y_pred = preds_map[best_model_name]
    best_y_prob = probs_map.get(best_model_name, None)

    # 7. Learning Curve pada model Tuned XGBoost
    lc_sizes, lc_train, lc_val = learning_curve(
        tuned_xgb,
        X_train_res,
        y_train_res,
        cv=3,
        scoring=scoring_name,
        train_sizes=np.linspace(0.25, 1.0, 4),
        shuffle=True,
        random_state=seed,
        n_jobs=-1,
    )

    # 8. Feature Importance dari Tuned XGBoost
    if hasattr(tuned_xgb, "feature_importances_"):
        importances = np.asarray(tuned_xgb.feature_importances_)
    else:
        importances = np.ones(len(selected_cols))

    imbalance_code = (
        "smote = SMOTE(random_state=GLOBAL_SEED)\n"
        "X_train, y_train = smote.fit_resample(X_train, y_train)"
        if imbalance_method == "smote" and task_type == "classification"
        else "# Menggunakan bobot kelas / data asli"
    )
    equivalent_code = f'''# Reproduksi Eksperimen Modul HARD (GLOBAL_SEED = {seed})
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb

GLOBAL_SEED = {seed}
selected_features = {selected_cols}
X = df[selected_features]
y = df["{target_col}"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=GLOBAL_SEED
)

# Penanganan Imbalanced Data ({imbalance_method.upper()})
{imbalance_code}

# Model Terbaik Hasil Hyperparameter Tuning ({tuning_method.upper()})
best_params = {best_params}
tuned_xgb = xgb.{"XGBClassifier" if task_type == "classification" else "XGBRegressor"}(
    **best_params, random_state=GLOBAL_SEED
)
tuned_xgb.fit(X_train, y_train)
y_pred = tuned_xgb.predict(X_test)
'''

    return {
        "leaderboard_df": leaderboard_df,
        "sort_col": sort_col,
        "best_model_name": best_model_name,
        "best_model_obj": best_model_obj,
        "best_params": best_params,
        "tuning_history_df": pd.DataFrame(tuning_history),
        "selected_cols": selected_cols,
        "all_cols": num_cols,
        "class_counts_before": class_counts_before,
        "class_counts_after": class_counts_after,
        "y_test": np.asarray(y_test),
        "best_y_pred": np.asarray(best_y_pred),
        "best_y_prob": best_y_prob,
        "lc_sizes": lc_sizes,
        "lc_train_mean": np.mean(lc_train, axis=1),
        "lc_val_mean": np.mean(lc_val, axis=1),
        "importances": importances,
        "scoring_name": scoring_name.upper(),
        "code": equivalent_code,
    }