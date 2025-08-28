import os, json, warnings
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, auc, brier_score_loss
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# Optional deps
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)

def _ensure_dirs():
    Path("assets").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    _ensure_dirs()
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    # synthesize minimal dataset (for first boot only)
    rng = np.random.default_rng(42)
    n = 180
    X = rng.normal(0, 1, size=(n, len(config.FEATURES)))
    y = (rng.random(n) > 0.5).astype(int)
    df = pd.DataFrame(X, columns=config.FEATURES); df[config.TARGET] = y
    df.insert(0, config.NAME_COL, [f"s{i:03d}" for i in range(n)])
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df

def validate_training_data(df: pd.DataFrame) -> Tuple[bool, list]:
    errs = []
    for col in config.FEATURES + [config.TARGET]:
        if col not in df.columns:
            errs.append(f"Missing column: {col}")
    if len(df) < 40:
        errs.append("Dataset too small (<40 rows).")
    return (len(errs) == 0, errs)

def _get_model_by_name(name: str, params: dict):
    name = name or config.DEFAULT_MODEL
    if name == "LogisticRegression":
        clf = LogisticRegression(**{k:v for k,v in params.items() if k in ["C","max_iter","penalty"]}, random_state=config.RANDOM_STATE)
    elif name == "RandomForest":
        clf = RandomForestClassifier(**{k:v for k,v in params.items() if k in ["n_estimators","max_depth","min_samples_split"]}, random_state=config.RANDOM_STATE, n_jobs=-1)
    elif name == "SVC":
        clf = SVC(**{k:v for k,v in params.items() if k in ["C","kernel","probability"]}, random_state=config.RANDOM_STATE)
    elif name == "GradientBoosting":
        clf = GradientBoostingClassifier(**{k:v for k,v in params.items() if k in ["n_estimators","learning_rate","max_depth"]}, random_state=config.RANDOM_STATE)
    elif name == "ExtraTrees":
        clf = ExtraTreesClassifier(**{k:v for k,v in params.items() if k in ["n_estimators","max_depth","min_samples_split"]}, random_state=config.RANDOM_STATE, n_jobs=-1)
    elif name == "XGBoost" and HAS_XGB:
        clf = xgb.XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate",0.05)),
            max_depth=int(params.get("max_depth",3)),
            subsample=float(params.get("subsample",0.9)),
            colsample_bytree=float(params.get("colsample_bytree",0.9)),
            eval_metric="logloss",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
            tree_method="hist",
        )
    elif name == "MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (128,64)),
            alpha=float(params.get("alpha", 0.0005)),
            max_iter=int(params.get("max_iter", 500)),
            random_state=config.RANDOM_STATE,
        )
    elif name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=int(params.get("n_neighbors",7)), weights=params.get("weights","distance"))
    elif name == "NaiveBayes":
        clf = GaussianNB()
    elif name == "DecisionTree":
        clf = DecisionTreeClassifier(max_depth=int(params.get("max_depth",5)), min_samples_split=int(params.get("min_samples_split",2)), random_state=config.RANDOM_STATE)
    elif name == "AdaBoost":
        clf = AdaBoostClassifier(n_estimators=int(params.get("n_estimators",200)), learning_rate=float(params.get("learning_rate",0.1)), random_state=config.RANDOM_STATE)
    elif name == "LDA":
        clf = LDA(solver=params.get("solver","svd"))
    elif name == "QDA":
        clf = QDA(reg_param=float(params.get("reg_param",0.0)))
    elif name == "Bagging":
        clf = BaggingClassifier(n_estimators=int(params.get("n_estimators",200)), max_samples=float(params.get("max_samples",1.0)), random_state=config.RANDOM_STATE, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=300, C=1.0, penalty="l2", random_state=config.RANDOM_STATE)

    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])
    return pipe

def _get_proba(pipe, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        if isinstance(proba, list):
            proba = np.asarray(proba)
        if proba.ndim == 1:
            return proba
        return proba[:,1]
    if hasattr(pipe, "decision_function"):
        z = pipe.decision_function(X)
        return 1/(1+np.exp(-z))
    return pipe.predict(X).astype(float)

def _opt_threshold(y_true, y_scores, mode: str = "youden"):
    if mode == "f1":
        prec, rec, thr = precision_recall_curve(y_true, y_scores)
        f1s = (2*prec*rec/(prec+rec+1e-9))
        i = int(np.nanargmax(f1s[:-1])) if len(f1s)>1 else 0
        t = thr[i] if len(thr)>0 else 0.5
        return float(t), {"f1_opt": float(np.nanmax(f1s))}
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    j = tpr - fpr
    i = int(np.nanargmax(j)) if len(j)>0 else 0
    t = thr[i] if len(thr)>0 else 0.5
    return float(t), {}

def _compute_metrics(y_true, y_scores, y_pred, model_name: str, thr_mode: str="youden"):
    m = {}
    m["roc_auc"] = float(roc_auc_score(y_true, y_scores))
    m["accuracy"] = float(accuracy_score(y_true, y_pred))
    m["f1"] = float(f1_score(y_true, y_pred))
    m["precision"] = float(precision_score(y_true, y_pred))
    m["recall"] = float(recall_score(y_true, y_pred))
    opt_thr, extra = _opt_threshold(y_true, y_scores, mode=thr_mode)
    m["opt_thr"] = float(opt_thr)
    m.update(extra)
    m["n_samples"] = int(len(y_true))
    return m

def _save_plots(y_true, y_scores, model_name: str, tag: str = "run"):
    assets = Path("assets"); assets.mkdir(parents=True, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC – {model_name}"); plt.legend()
    roc_path = assets / f"roc_{tag}.png"; plt.savefig(roc_path, dpi=150, bbox_inches="tight"); plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_scores); ap = average_precision_score(y_true, y_scores)
    plt.figure(); plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR – {model_name}"); plt.legend()
    pr_path = assets / f"pr_{tag}.png"; plt.savefig(pr_path, dpi=150, bbox_inches="tight"); plt.close()
    # CM
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, (y_scores>=0.5).astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm); disp.plot(values_format="d")
    plt.title("Confusion Matrix"); cm_path = assets / f"cm_{tag}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close()
    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy="quantile")
    plt.figure(); plt.plot([0,1],[0,1],"--"); plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Mean predicted prob"); plt.ylabel("Fraction of positives"); plt.title(f"Calibration – {model_name}")
    cal_path = assets / f"cal_{tag}.png"; plt.savefig(cal_path, dpi=150, bbox_inches="tight"); plt.close()
    return {
        "roc": {"fpr": list(map(float,fpr)), "tpr": list(map(float,tpr)), "auc": float(roc_auc), "path": str(roc_path)},
        "pr": {"prec": list(map(float,prec)), "rec": list(map(float,rec)), "ap": float(ap), "path": str(pr_path)},
        "cm": {"matrix": cm.tolist(), "path": str(cm_path)},
        "cal": {"path": str(cal_path)}
    }

def create_pipeline(model_name: str, model_params: dict):
    return _get_model_by_name(model_name, model_params or {})

def train_model(data_path: str, model_name: str, model_params: dict,
                test_size: float=0.2, do_cv: bool=True, do_tune: bool=True,
                artifact_tag: str = "run", thr_mode: str = "youden"):
    _ensure_dirs()
    df = load_data(data_path)
    ok, errs = validate_training_data(df)
    if not ok:
        return {"ok": False, "errors": errs}

    X = df[config.FEATURES]; y = df[config.TARGET].astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE
    )

    pipe = create_pipeline(model_name, model_params)

    cv_means = None
    if do_cv:
        try:
            cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
            scoring = ["roc_auc","accuracy","f1","precision","recall"]
            cv_scores = cross_validate(
                pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1,
                return_train_score=False, error_score=np.nan
            )
            cv_means = {m: float(np.nanmean(cv_scores.get(f"test_{m}", np.array([np.nan]))))
                        for m in ["roc_auc","accuracy","f1","precision","recall"]}
        except Exception:
            cv_means = None

    if do_tune:
        try:
            grid = config.PARAM_GRIDS.get(model_name, None)
            if grid:
                gs = GridSearchCV(pipe, grid, scoring=config.SCORING, cv=3, n_jobs=-1, refit=True, error_score=float("nan"))
                gs.fit(X_tr, y_tr)
                pipe = gs.best_estimator_
        except Exception:
            pass

    pipe.fit(X_tr, y_tr)
    y_scores = _get_proba(pipe, X_val)
    thr = 0.5
    if isinstance(y_scores, np.ndarray):
        thr, _ = _opt_threshold(y_val, y_scores, mode=thr_mode)
    y_pred = (y_scores>=thr).astype(int)
    metrics = _compute_metrics(y_val, y_scores, y_pred, model_name, thr_mode=thr_mode)
    curves = _save_plots(y_val, y_scores, model_name, tag=artifact_tag)

    # Permutation importance (best-effort)
    try:
        pi = permutation_importance(pipe, X_val, y_val, n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1)
        importances = pd.DataFrame({"feature": list(X_val.columns), "importance": pi.importances_mean}).sort_values("importance", ascending=False).head(10)
        importances.to_csv(Path("assets")/f"permimp_{artifact_tag}.csv", index=False)
        plt.figure(); plt.barh(importances["feature"][::-1], importances["importance"][::-1])
        plt.title(f"Permutation importance – {model_name}"); plt.tight_layout()
        pim_path = Path("assets")/f"permimp_{artifact_tag}.png"; plt.savefig(pim_path, dpi=150, bbox_inches="tight"); plt.close()
        curves["permimp"] = {"path": str(pim_path)}
    except Exception:
        pass

    cand_path = f"models/candidate_{artifact_tag}.joblib"
    joblib.dump(pipe, cand_path)

    try:
        import datetime as _dt
        runs = Path(config.RUNS_CSV)
        runs.parent.mkdir(parents=True, exist_ok=True)
        row = {"tag":artifact_tag,"model":model_name,"roc_auc":metrics["roc_auc"],"f1":metrics["f1"],"accuracy":metrics["accuracy"],
               "precision":metrics["precision"],"recall":metrics["recall"],"opt_thr":metrics.get("opt_thr",0.5),
               "time":_dt.datetime.utcnow().isoformat()}
        if runs.exists():
            df_runs = pd.read_csv(runs); df_runs = pd.concat([df_runs, pd.DataFrame([row])], ignore_index=True)
        else:
            df_runs = pd.DataFrame([row])
        df_runs.to_csv(runs, index=False)
    except Exception:
        pass

    return {
        "ok": True,
        "candidate_path": cand_path,
        "val_metrics": metrics,
        "cv_means": cv_means,
        "curves": curves,
        "params_used": model_params
    }

def has_production() -> bool:
    return Path(config.MODEL_PATH).exists()

def read_best_meta() -> Dict[str, Any]:
    p = Path("assets/best_model.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def promote_model_to_production(candidate_path: str, metadata: Dict[str, Any] = None) -> str:
    _ensure_dirs()
    dst = Path(config.MODEL_PATH); dst.parent.mkdir(parents=True, exist_ok=True)
    if not Path(candidate_path).exists():
        raise FileNotFoundError(f"Candidate not found: {candidate_path}")
    import shutil; shutil.copyfile(candidate_path, dst)
    meta = read_best_meta()
    meta.update(metadata or {})
    if "opt_thr" not in meta:
        meta["opt_thr"] = 0.5
    Path("assets/best_model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"Promoted {candidate_path} → {dst}"

def _predict_core(model_path: str, X: pd.DataFrame, threshold: float=None) -> pd.DataFrame:
    pipe = joblib.load(model_path)
    scores = _get_proba(pipe, X)
    thr = 0.5 if threshold is None else float(threshold)
    pred = (scores>=thr).astype(int)
    return pd.DataFrame({"proba_PD": scores, "pred": pred})

def predict_with_production(X: pd.DataFrame, threshold: float=None) -> pd.DataFrame:
    if not has_production(): raise FileNotFoundError("No production model found.")
    meta = read_best_meta()
    thr = threshold if threshold is not None else float(meta.get("opt_thr", 0.5))
    return _predict_core(config.MODEL_PATH, X, threshold=thr)

def run_prediction(row_df: pd.DataFrame) -> Tuple[int, float]:
    out = predict_with_production(row_df)
    return int(out.iloc[0]["pred"]), float(out.iloc[0]["proba_PD"])

def batch_predict_frame(df_in: pd.DataFrame) -> pd.DataFrame:
    feats = config.FEATURES
    X = df_in[feats] if all(f in df_in.columns for f in feats) else df_in
    preds = predict_with_production(X)
    return pd.concat([df_in.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

def evaluate_model(model_path: str, data_path: str=None, artifact_tag: str="best_eval") -> Dict[str, Any]:
    df = load_data(data_path or config.DATA_PATH)
    X = df[config.FEATURES]; y = df[config.TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )
    pipe = joblib.load(model_path)
    scores = _get_proba(pipe, X_te)
    meta = read_best_meta()
    thr = float(meta.get("opt_thr", 0.5))
    y_pred = (scores>=thr).astype(int)
    metrics = _compute_metrics(y_te, scores, y_pred, "production", thr_mode="youden")
    curves = _save_plots(y_te, scores, "production", tag=artifact_tag)
    # Calibration loss
    try:
        metrics["brier_score"] = float(brier_score_loss(y_te, scores))
    except Exception:
        pass
    # SameFileError fix
    if artifact_tag.startswith("best_eval"):
        import shutil
        pairs = [
            (curves["roc"]["path"], "assets/roc_best_eval.png"),
            (curves["pr"]["path"], "assets/pr_best_eval.png"),
            (curves["cm"]["path"], "assets/cm_best_eval.png"),
            (curves.get("cal",{}).get("path",""), "assets/cal_best_eval.png"),
        ]
        for src, dst in pairs:
            if not src:
                continue
            sp, dp = Path(src), Path(dst)
            if sp.exists():
                try:
                    if sp.resolve() == dp.resolve():
                        continue
                except Exception:
                    if str(sp) == str(dp):
                        continue
                shutil.copyfile(sp, dp)
    return {"metrics": metrics, "curves": curves}

def _fit_and_score_for_compare(data_path: str, model_name: str, params: dict, thr_mode: str="youden") -> Dict[str, Any]:
    res = train_model(data_path, model_name, params, do_cv=True, do_tune=True, artifact_tag=f"eda_{model_name}", thr_mode=thr_mode)
    return {"name": model_name, "metrics": res["val_metrics"], "cand": res["candidate_path"], "curves": res["curves"]}

def auto_init_production_from_eda() -> Dict[str, Any]:
    _ensure_dirs()
    if has_production():
        return {"initialized": False, "reason": "already_present"}
    df = load_data(config.DATA_PATH)
    ok, errs = validate_training_data(df)
    if not ok:
        raise RuntimeError("Bad data for auto-init: " + "; ".join(errs))
    leaderboard = []
    for m in config.EDA_MODELS:
        params = config.DEFAULT_PARAMS.get(m, {})
        out = _fit_and_score_for_compare(config.DATA_PATH, m, params, thr_mode="youden")
        leaderboard.append(out)
    best = max(leaderboard, key=lambda d: float(d["metrics"]["roc_auc"]))
    meta = {"source":"auto_init_from_eda","model_name":best["name"], "opt_thr": float(best["metrics"].get("opt_thr",0.5))}
    promote_model_to_production(best["cand"], metadata=meta)
    import pandas as pd
    rows = []
    for d in leaderboard:
        r = {"model": d["name"]}; r.update(d["metrics"]); rows.append(r)
    pd.DataFrame(rows).to_csv("assets/eda_leaderboard.csv", index=False)
    return {"initialized": True, "best_model": best["name"], "best_auc": float(best["metrics"]["roc_auc"])}
