
import streamlit as st
import numpy as np, pandas as pd, os

st.set_page_config(page_title="Log Anomaly Dashboard", layout="wide")
st.title("Unsupervised Log Anomaly Detection (Advanced)")

if st.sidebar.button("Reload Artifacts"):
    if os.path.exists("artifacts/scores.npy"):
        scores = np.load("artifacts/scores.npy")
        st.metric("Total Events", len(scores))
        st.metric("Top 1% Threshold", f"{np.quantile(scores, 0.99):.3f}")
        st.line_chart(pd.Series(scores).rolling(50).mean())
    else:
        st.warning("Run evaluation first to produce artifacts/*.npy")

st.subheader("Top Anomalies")
if os.path.exists("artifacts/top_anomalies.txt"):
    st.code(open("artifacts/top_anomalies.txt").read()[:5000], language="text")
else:
    st.info("No anomalies file yet. Run evaluate script.")
