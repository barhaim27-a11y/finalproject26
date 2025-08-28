import io, json
import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config
import model_pipeline as mp

st.set_page_config(page_title="Parkinsons – Pro (v8.9)", layout="wide")
st.title("🧠 Parkinsons – ML App (Pro, v8.9)")
st.caption("Production is auto-selected from YOUR notebook's EDA models on first run (no uploads). Replace via Retrain → Promote.")

def to_excel_bytes(sheets: dict) -> bytes:
    bio = io.BytesIO()
    try:
        try: import openpyxl; engine="openpyxl"
        except Exception: import xlsxwriter; engine="xlsxwriter"
        with pd.ExcelWriter(bio, engine=engine) as writer:
            for name, df in sheets.items():
                if not isinstance(df, pd.DataFrame): df = pd.DataFrame(df)
                df.to_excel(writer, sheet_name=(name or "Sheet")[:31], index=False)
        bio.seek(0); return bio.read()
    except Exception:
        first_df = next(iter(sheets.values())) if sheets else pd.DataFrame()
        return first_df.to_csv(index=False).encode("utf-8")

def read_csv_flex(file) -> pd.DataFrame:
    for enc in ["utf-8","latin-1","cp1255"]:
        try: file.seek(0); return pd.read_csv(file, encoding=enc)
        except Exception: continue
    file.seek(0); return pd.read_csv(file, errors="ignore")

# Auto-init production if missing (train all detected EDA models and pick best)
try:
    info = mp.auto_init_production_from_eda()
    if info.get("initialized"):
        st.info(f"Auto-initialized Production from EDA: {info['best_model']} (ROC-AUC={info['best_auc']:.3f})")
except Exception as e:
    st.warning(f"Auto-init failed: {e}")

df = mp.load_data(config.DATA_PATH)
features = config.FEATURES; target = config.TARGET

# Helper: risk labels
LOW, HIGH = float(config.RISK_THRESHOLDS.get("low",0.3)), float(config.RISK_THRESHOLDS.get("high",0.7))
def risk_bucket(p: float):
    if p < LOW: return "סיכון נמוך", "🟢", "low"
    if p < HIGH: return "סיכון בינוני", "🟡", "medium"
    return "סיכון גבוה", "🔴", "high"

def explain_sentence(prob: float, thr: float):
    label = "PD" if prob >= thr else "No-PD"
    risk, emoji, _ = risk_bucket(prob)
    return f"{emoji} **{risk}** — ההסתברות היא **{prob:.1%}**. עם סף החלטה **{thr:.2f}** הסיווג הוא **{label}**."

tab_data, tab_single, tab_multi, tab_best, tab_predict, tab_retrain = st.tabs(
    ["DATA / EDA","Single Model","Multi Compare","Best Dashboard","Predict","Retrain"]
)

with tab_data:
    st.subheader("Dataset")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(30), use_container_width=True)

    st.markdown("### EDA – Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        miss_df = df[features + [target]].isna().sum().sort_values(ascending=False).rename("missing").reset_index().rename(columns={"index":"column"})
        st.write("Missing:"); st.dataframe(miss_df, use_container_width=True)
        st.download_button("missing.csv", miss_df.to_csv(index=False), "missing.csv", "text/csv")
    with c2:
        desc_df = df[features].describe().T
        st.write("Describe:"); st.dataframe(desc_df, use_container_width=True)
        st.download_button("describe.csv", desc_df.to_csv(), "describe.csv", "text/csv")
    with c3:
        cls = df[target].value_counts().rename({0:"No-PD",1:"PD"})
        st.write("Class balance:"); st.bar_chart(cls)

    st.markdown("### Distributions (selected features)")
    cols = st.columns(3)
    for i, feat in enumerate(features[:9]):
        with cols[i%3]:
            fig, ax = plt.subplots(figsize=(3,2))
            ax.hist(df[feat].dropna(), bins=20)
            ax.set_title(feat); st.pyplot(fig)

    st.markdown("### Box / Violin by Class")
    cols2 = st.columns(2)
    for i, feat in enumerate(features[:6]):
        with cols2[i%2]:
            fig, ax = plt.subplots(figsize=(4,3))
            data0 = df[df[target]==0][feat].dropna(); data1 = df[df[target]==1][feat].dropna()
            ax.boxplot([data0, data1], labels=["No-PD","PD"])
            ax.set_title(f"Box – {feat}"); st.pyplot(fig)

    st.markdown("### Correlation heatmap")
    corr = df[features + [target]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.imshow(corr.values)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

    st.markdown("### PCA (2D)")
    try:
        from sklearn.decomposition import PCA
        X = df[features].select_dtypes(include=[float,int]).fillna(df[features].median(numeric_only=True))
        X = (X - X.mean())/X.std(ddof=0)
        pc = PCA(n_components=2, random_state=42).fit_transform(X.values)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        colors = np.where(df[target]==1, 1, 0)
        ax2.scatter(pc[:,0], pc[:,1], c=colors, cmap="coolwarm", s=25, alpha=0.7)
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_title("PCA Projection")
        st.pyplot(fig2)
    except Exception as e:
        st.info(f"PCA skipped: {e}")

with tab_single:
    st.subheader("Train & Evaluate a single model (visual only)")
    model_options = config.MODEL_LIST_SINGLE
    chosen = st.selectbox("Model", model_options, index=model_options.index(config.DEFAULT_MODEL))
    colA, colB = st.columns(2)
    with colA: do_cv = st.checkbox("Cross-Validation", True)
    with colB: do_tune = st.checkbox("GridSearch", True)
    thr_mode = st.selectbox("Optimal threshold method", ["youden","f1"], index=0)

    def edit_params(model_name: str, key_prefix: str=""):
        params = config.DEFAULT_PARAMS.get(model_name, {}).copy()
        cols = st.columns(3); edited={}; i=0
        for k,v in params.items():
            with cols[i%3]:
                skey = f"{key_prefix}{model_name}_{k}"
                if isinstance(v,bool): edited[k]=st.checkbox(k,value=v,key=skey)
                elif isinstance(v,int): edited[k]=st.number_input(k,value=int(v),step=1,key=skey)
                elif isinstance(v,float): edited[k]=st.number_input(k,value=float(v),key=skey,format="%.6f")
                elif isinstance(v,tuple): edited[k]=st.text_input(k,value=str(v),key=skey)
                else: edited[k]=st.text_input(k,value=str(v),key=skey)
            i+=1
        for k,v in edited.items():
            if isinstance(v,str) and v.startswith("(") and v.endswith(")"):
                try: edited[k]=eval(v)
                except Exception: pass
        return edited

    params = edit_params(chosen, "single_")
    if st.button("Train model", key="single_train"):
        res = mp.train_model(config.DATA_PATH, chosen, params, do_cv=do_cv, do_tune=do_tune,
                             artifact_tag=f"single_{chosen}", thr_mode=thr_mode)
        if not res.get("ok"):
            st.error("\n".join(res.get("errors", [])))
        else:
            st.success(f"Candidate saved: {res['candidate_path']} (visual only)")
            mets = pd.DataFrame([res["val_metrics"]]); st.dataframe(mets, use_container_width=True)
            if res.get("cv_means"):
                st.markdown("**Cross-Validation (means):**"); st.dataframe(pd.DataFrame([res["cv_means"]]), use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            from pathlib import Path
            roc_p = Path(res["curves"]["roc"]["path"]); pr_p = Path(res["curves"]["pr"]["path"]); cm_p = Path(res["curves"]["cm"]["path"]); cal_p = Path(res["curves"].get("cal",{}).get("path",""))
            if roc_p.exists(): c1.image(str(roc_p), caption=f"ROC – AUC={res['curves']['roc']['auc']:.3f}")
            if pr_p.exists(): c2.image(str(pr_p), caption=f"PR – AP={res['curves']['pr']['ap']:.3f}")
            if cm_p.exists(): c3.image(str(cm_p), caption="Confusion Matrix")
            if cal_p and cal_p.exists(): c4.image(str(cal_p), caption="Calibration")
            pim = Path(res["curves"].get("permimp",{}).get("path",""))
            if pim and pim.exists(): st.image(str(pim), caption="Permutation Importance (Top 10)")
            xls = to_excel_bytes({"metrics": mets, "cv_means": pd.DataFrame([res["cv_means"]]) if res.get("cv_means") else pd.DataFrame()})
            st.download_button("Export Single Results (XLSX)", xls, f"single_{chosen}_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="single_xlsx")

with tab_multi:
    st.subheader("Train & Compare multiple models (from notebook's EDA set + extras)")
    pick = st.multiselect("Select models", options=config.MODEL_LIST_MULTI, default=config.EDA_MODELS[:min(3,len(config.EDA_MODELS))])
    do_cv2 = st.checkbox("Cross-Validation", True, key="multi_cv")
    do_tune2 = st.checkbox("GridSearch", True, key="multi_tune")
    thr_mode2 = st.selectbox("Threshold mode", ["youden","f1"], index=0, key="multi_thr")

    def edit_params_m(model_name: str, key_prefix: str=""):
        params = config.DEFAULT_PARAMS.get(model_name, {}).copy()
        cols = st.columns(3); edited={}; i=0
        for k,v in params.items():
            with cols[i%3]:
                skey = f"{key_prefix}{model_name}_{k}"
                if isinstance(v,bool): edited[k]=st.checkbox(k,value=v,key=skey)
                elif isinstance(v,int): edited[k]=st.number_input(k,value=int(v),step=1,key=skey)
                elif isinstance(v,float): edited[k]=st.number_input(k,value=float(v),key=skey,format="%.6f")
                elif isinstance(v,tuple): edited[k]=st.text_input(k,value=str(v),key=skey)
                else: edited[k]=st.text_input(k,value=str(v),key=skey)
            i+=1
        for k,v in edited.items():
            if isinstance(v,str) and v.startswith("(") and v.endswith(")"):
                try: edited[k]=eval(v)
                except Exception: pass
        return edited

    param_map={}
    for m in pick:
        with st.expander(f"Parameters – {m}", expanded=False):
            param_map[m] = edit_params_m(m, f"multi_{m}_")

    if st.button("Train & Compare", key="multi_train"):
        leaderboard=[]; curves={}
        for m in pick:
            res = mp.train_model(config.DATA_PATH, m, param_map.get(m, {}), do_cv=do_cv2, do_tune=do_tune2,
                                 artifact_tag=f"multi_{m}", thr_mode=thr_mode2)
            if res.get("ok"):
                row = res["val_metrics"].copy(); row["model_name"]=m; leaderboard.append(row)
                curves[m]=res["curves"]
        if leaderboard:
            df_lb = pd.DataFrame(leaderboard).sort_values("roc_auc", ascending=False).reset_index(drop=True)
            i_best = df_lb["roc_auc"].astype(float).idxmax()
            df_lb.loc[i_best, "model_name"] = "⭐ " + str(df_lb.loc[i_best, "model_name"])
            st.dataframe(df_lb, use_container_width=True)
            metric_choice = st.selectbox("Metric for bar chart", ["roc_auc","accuracy","f1","precision","recall"], index=0)
            st.bar_chart(pd.DataFrame(df_lb.set_index("model_name")[metric_choice]))
            figR, axR = plt.subplots(figsize=(6,4))
            for name,c in curves.items(): axR.plot(c["roc"]["fpr"], c["roc"]["tpr"], label=name)
            axR.plot([0,1],[0,1],"--", lw=0.7); axR.set_xlabel("FPR"); axR.set_ylabel("TPR"); axR.legend(); axR.set_title("ROC")
            st.pyplot(figR)
            figP, axP = plt.subplots(figsize=(6,4))
            for name,c in curves.items(): axP.plot(c["pr"]["rec"], c["pr"]["prec"], label=name)
            axP.set_xlabel("Recall"); axP.set_ylabel("Precision"); axP.legend(); axP.set_title("PR")
            st.pyplot(figP)
            xls = to_excel_bytes({"leaderboard": df_lb})
            st.download_button("Export Leaderboard (XLSX)", xls, "leaderboard.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="multi_xlsx")
            st.download_button("leaderboard.csv", df_lb.to_csv(index=False), "leaderboard.csv", "text/csv", key="multi_csv")
        else:
            st.warning("No models trained.")

with tab_best:
    st.subheader("Production (auto-selected from notebook EDA) – Dashboard")
    prod = Path(config.MODEL_PATH)
    if not prod.exists():
        st.error("Production model is missing (unexpected). Try rerun.")
    else:
        colB1, colB2 = st.columns([2,1])
        with colB1:
            ev = mp.evaluate_model(str(prod), artifact_tag="best_eval")
            mets = pd.DataFrame([ev["metrics"]])
            st.dataframe(mets, use_container_width=True)
            for p,cap in [("assets/roc_best_eval.png","ROC"),("assets/pr_best_eval.png","PR"),("assets/cm_best_eval.png","Confusion Matrix"),("assets/cal_best_eval.png","Calibration")]:
                if Path(p).exists(): st.image(p, caption=cap)
            st.download_button("best_eval_metrics.csv", mets.to_csv(index=False), "best_eval_metrics.csv", "text/csv")
        with colB2:
            st.markdown("**Metadata**")
            st.json(mp.read_best_meta())
            st.info("This model was chosen automatically from your notebook's EDA models on first run. Replace via Retrain → Promote.")

with tab_predict:
    st.subheader("Predict with Production (single & batch)")
    if not Path(config.MODEL_PATH).exists():
        st.error("Missing Production model. Please rerun app.")
    else:
        meta = mp.read_best_meta()
        default_thr = float(meta.get("opt_thr", 0.5))
        thr = st.slider("Decision threshold", 0.0, 1.0, value=float(default_thr), step=0.01)
        st.caption(f"סף ברירת מחדל (מהמטא־דאטה): {default_thr:.2f}")

        preds = mp.predict_with_production(df[features], threshold=thr)
        df_full = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

        # Add friendly labels
        risk_labels, decisions = [], []
        for p in df_full["proba_PD"].astype(float):
            risk, emoji, _ = risk_bucket(p)
            risk_labels.append(f"{emoji} {risk}")
            decisions.append("PD" if p >= thr else "No-PD")
        df_full["risk_label"] = risk_labels
        df_full["decision"] = decisions

        st.markdown("**Predictions preview:**")
        st.dataframe(df_full.head(30), use_container_width=True)

        figH, axH = plt.subplots(figsize=(6,3))
        axH.hist(df_full["proba_PD"], bins=20)
        axH.axvline(thr, linestyle="--")
        axH.set_xlabel("Predicted probability of PD"); axH.set_ylabel("Count"); axH.set_title("Probability distribution vs. threshold")
        st.pyplot(figH)

        xls = to_excel_bytes({"predictions": df_full})
        st.download_button("Export predictions (XLSX)", xls, "predictions.xlsx", key="pred_xlsx")
        st.download_button("predictions.csv", df_full.to_csv(index=False), "predictions.csv", "text/csv", key="pred_csv")

        st.markdown("### Predict single case")
        cols = st.columns(3); single={}
        for i,f in enumerate(features):
            with cols[i%3]:
                default = float(df[f].median()) if f in df.columns else 0.0
                single[f] = st.number_input(f, value=default, format="%.6f", key=f"single_{f}")
        if st.button("Predict single"):
            row = pd.DataFrame([single])
            out = mp.predict_with_production(row, threshold=thr)
            prob = float(out.iloc[0]["proba_PD"])
            pred = int(out.iloc[0]["pred"])
            st.success(explain_sentence(prob, thr))
            st.progress(min(max(prob, 0.0), 1.0), text=f"Probability: {prob:.1%}")
            with st.expander("מה זה אומר?"):
                st.write(
                    "- **הסתברות** היא מד ביטחון של המודל, לא אבחנה רפואית.\n"
                    "- **סף החלטה** קובע חיובי/שלילי; הורדתו מעלה רגישות, העלאתו מעלה דיוק חיובי.\n"
                    "- מומלץ לעיין גם ב־**Best Dashboard** (ROC/PR/CM/Calibration)."
                )

        st.markdown("### Batch prediction (CSV)")
        up = st.file_uploader("Upload CSV (features only)", type=["csv"], key="pred_batch")
        if st.button("Run batch predictions"):
            if up is None:
                st.error("Please upload a CSV.")
            else:
                df_in = read_csv_flex(up)
                preds_b = mp.predict_with_production(df_in, threshold=thr)
                df_batch = pd.concat([df_in.reset_index(drop=True), preds_b.reset_index(drop=True)], axis=1)
                risk_labels, decisions, explain = [], [], []
                for p in df_batch["proba_PD"].astype(float):
                    risk, emoji, _ = risk_bucket(p)
                    risk_labels.append(f"{emoji} {risk}")
                    decisions.append("PD" if p >= thr else "No-PD")
                    explain.append(explain_sentence(p, thr))
                df_batch["risk_label"] = risk_labels
                df_batch["decision"] = decisions
                df_batch["explanation"] = explain
                st.dataframe(df_batch.head(30), use_container_width=True)
                xlsb = to_excel_bytes({"batch_predictions": df_batch})
                st.download_button("Export batch predictions (XLSX)", xlsb, "batch_predictions.xlsx", key="pred_batch_xlsx")
                st.download_button("batch_predictions.csv", df_batch.to_csv(index=False), "batch_predictions.csv", "text/csv", key="pred_batch_csv")

with tab_retrain:
    st.subheader("Retrain with new data → Compare vs. Production → Promote if better")
    up_new = st.file_uploader("Upload training CSV (same schema: features + status)", type=["csv"], key="train_new")
    metric_for_promotion = st.selectbox("Metric", ["roc_auc","f1","accuracy","precision","recall"], index=0)

    st.markdown("**Retrain a single model**")
    model_r = st.selectbox("Model", config.MODEL_LIST_RETRAIN, index=config.MODEL_LIST_RETRAIN.index(config.DEFAULT_MODEL), key="re_model")

    def edit_params_r(model_name: str, key_prefix: str=""):
        params = config.DEFAULT_PARAMS.get(model_name, {}).copy()
        cols = st.columns(3); edited={}; i=0
        for k,v in params.items():
            with cols[i%3]:
                skey = f"{key_prefix}{model_name}_{k}"
                if isinstance(v,bool): edited[k]=st.checkbox(k,value=v,key=skey)
                elif isinstance(v,int): edited[k]=st.number_input(k,value=int(v),step=1,key=skey)
                elif isinstance(v,float): edited[k]=st.number_input(k,value=float(v),key=skey,format="%.6f")
                elif isinstance(v,tuple): edited[k]=st.text_input(k,value=str(v),key=skey)
                else: edited[k]=st.text_input(k,value=str(v),key=skey)
            i+=1
        for k,v in edited.items():
            if isinstance(v,str) and v.startswith("(") and v.endswith(")"):
                try: edited[k]=eval(v)
                except Exception: pass
        return edited

    params_r = edit_params_r(model_r, "re_")
    thr_mode_r = st.selectbox("Threshold mode", ["youden","f1"], index=0, key="re_thr")
    if st.button("Train single on uploaded data"):
        if up_new is None:
            st.error("Please upload a CSV.")
        else:
            df_new = read_csv_flex(up_new)
            tmp_path = "data/_uploaded_train.csv"; df_new.to_csv(tmp_path, index=False)
            res_new = mp.train_model(tmp_path, model_name=model_r, model_params=params_r, do_cv=True, do_tune=True, artifact_tag=f"upload_{model_r}", thr_mode=thr_mode_r)
            if not res_new.get("ok"):
                st.error("\n".join(res_new.get("errors", [])))
            else:
                st.success("New candidate trained.")
                st.json(res_new["val_metrics"])
                can_promote=False
                if mp.has_production():
                    ev_prod = mp.evaluate_model(config.MODEL_PATH, data_path=tmp_path, artifact_tag="prod_eval")
                    st.write("Production metrics on the same uploaded data:"); st.json(ev_prod["metrics"])
                    new_v = float(res_new["val_metrics"].get(metric_for_promotion, float("-inf")))
                    old_v = float(ev_prod["metrics"].get(metric_for_promotion, float("-inf")))
                    can_promote = new_v > old_v
                    (st.success if can_promote else st.info)(f"Better on {metric_for_promotion}: {new_v:.4f} {'>' if can_promote else '≤'} {old_v:.4f}")
                else:
                    st.info("No Production model yet; you can promote now.")
                    can_promote = True
                if st.button("Promote this candidate", disabled=not can_promote):
                    meta = {"source":"retrain_single","model_name":model_r, "metrics":res_new["val_metrics"], "params":res_new.get("params_used", params_r)}
                    if "opt_thr" in res_new["val_metrics"]: meta["opt_thr"] = float(res_new["val_metrics"]["opt_thr"])
                    msg = mp.promote_model_to_production(res_new["candidate_path"], metadata=meta)
                    st.success(msg)

    st.markdown("---")
    st.markdown("**Train & Compare multiple models**")
    pick_r = st.multiselect("Models", config.MODEL_LIST_RETRAIN, default=config.EDA_MODELS, key="re_pick")
    thr_mode_m = st.selectbox("Threshold mode (multi)", ["youden","f1"], index=0, key="re_multi_thr")
    param_map_r = {}
    for m in pick_r:
        with st.expander(f"Parameters – {m} (retrain)", expanded=False):
            param_map_r[m] = edit_params_r(m, f"re_multi_{m}_")
    if st.button("Train & Compare (uploaded data)", key="re_multi_btn"):
        if up_new is None:
            st.error("Please upload a CSV.")
        else:
            df_new = read_csv_flex(up_new)
            tmp_path = "data/_uploaded_train.csv"; df_new.to_csv(tmp_path, index=False)
            leaderboard=[]; curves={}; cand_map={}
            for m in pick_r:
                res = mp.train_model(tmp_path, model_name=m, model_params=param_map_r.get(m, {}), do_cv=True, do_tune=True, artifact_tag=f"upload_{m}", thr_mode=thr_mode_m)
                if res.get("ok"):
                    row = res["val_metrics"].copy(); row["model_name"]=m; leaderboard.append(row)
                    curves[m]=res["curves"]; cand_map[m]=res["candidate_path"]
            if leaderboard:
                df_lb = pd.DataFrame(leaderboard).sort_values("roc_auc", ascending=False).reset_index(drop=True)
                i_best = df_lb["roc_auc"].astype(float).idxmax()
                df_lb.loc[i_best, "model_name"] = "⭐ " + str(df_lb.loc[i_best, "model_name"])
                st.dataframe(df_lb, use_container_width=True)
                can_promote=False; top_name = df_lb.iloc[0]["model_name"].replace("⭐ ","")
                if mp.has_production():
                    ev_prod = mp.evaluate_model(config.MODEL_PATH, data_path=tmp_path, artifact_tag="prod_eval_multi")
                    new_v = float(df_lb.iloc[0]["roc_auc"]); old_v = float(ev_prod["metrics"].get("roc_auc", float("-inf")))
                    can_promote = new_v > old_v
                    (st.success if can_promote else st.info)(f"Top is better on roc_auc: {new_v:.4f} {'>' if can_promote else '≤'} {old_v:.4f}")
                else:
                    st.info("No Production model yet; you can promote the top candidate.")
                    can_promote = True
                figR, axR = plt.subplots(figsize=(6,4))
                for name,c in curves.items(): axR.plot(c["roc"]["fpr"], c["roc"]["tpr"], label=name)
                axR.plot([0,1],[0,1],"--", lw=0.7); axR.set_xlabel("FPR"); axR.set_ylabel("TPR"); axR.legend(); axR.set_title("ROC")
                st.pyplot(figR)
                figP, axP = plt.subplots(figsize=(6,4))
                for name,c in curves.items(): axP.plot(c["pr"]["rec"], c["pr"]["prec"], label=name)
                axP.set_xlabel("Recall"); axP.set_ylabel("Precision"); axP.legend(); axP.set_title("PR")
                st.pyplot(figP)
                xls = to_excel_bytes({"retrain_leaderboard": df_lb})
                st.download_button("Export retrain leaderboard (XLSX)", xls, "retrain_leaderboard.xlsx", key="re_multi_xlsx")
                if st.button(f"Promote top candidate ({top_name})", disabled=not can_promote):
                    msg = mp.promote_model_to_production(cand_map[top_name], metadata={"source":"retrain_multi","model_name":top_name})
                    st.success(msg)
            else:
                st.warning("No models trained.")

st.markdown("---")
st.caption("v8.9 • Friendly risk labels • README updated • Extra EDA models (LDA/QDA/Bagging) • Predict uses Production • Replace via Retrain → Promote")
