
# Unsupervised-Log-Anomaly-Detection-Microservices-Advanced

**Unsupervised Anomaly Detection for Log Data in Microservice Architectures — Advanced Version**

This repository provides a research-grade, unsupervised anomaly detection framework for microservice logs, combining multi-modal embeddings, graph attention networks, hybrid ensembles with GMM thresholding, streaming simulation, and explainability (SHAP + gradients). A minimal Streamlit dashboard is included for real-time visualization.

## Key Additions Over the Basic Version
- Multi-modal log representations: Transformer semantics + LSTM temporal encoder + statistical frequency features, fused with an attention-gating layer.
- Dynamic service graph with Graph Attention Network (GAT) for cross‑service dependency modeling.
- Hybrid anomaly ensemble: Autoencoder reconstruction + IsolationForest + GMM-based dynamic thresholding → probabilistic anomaly score.
- Streaming simulator to emulate real microservice logs with controllable anomaly injection.
- Explainability: SHAP values on fused features; token-level gradient saliency for the Transformer branch.
- Streamlit dashboard: Real-time anomaly counts, 2D embeddings, and top contributing tokens/features.
- CI + tests: Import tests, basic trainer tests, and style checks.

## Repository Layout
```
.
├─ src/
│  ├─ data/
│  │   ├─ parser.py               # Log parsing & templating
│  │   ├─ dataset.py              # Datasets & tokenization
│  │   └─ graph_builder.py        # Build service dependency graphs
│  ├─ models/
│  │   ├─ transformer.py          # Transformer encoder (semantic branch)
│  │   ├─ lstm.py                 # LSTM temporal branch
│  │   ├─ fusion.py               # Feature fusion with attention gating
│  │   ├─ gnn_gat.py              # Graph Attention Network (GAT)
│  │   ├─ autoencoder.py          # Reconstruction head
│  │   └─ ensemble.py             # IsolationForest + GMM thresholding
│  ├─ explain/
│  │   ├─ shap_explain.py         # SHAP explainability for fused features
│  │   └─ grad_tokens.py          # Token-level gradient saliency
│  ├─ streaming/
│  │   └─ collector.py            # Log stream simulator
│  ├─ train.py                    # Unsupervised training pipeline
│  ├─ evaluate.py                 # Evaluation & metrics + thresholding
│  ├─ visualize.py                # Embedding/graph plots
│  └─ dashboard_app.py            # Streamlit dashboard
├─ configs/
│  ├─ base.yaml
│  └─ advanced.yaml
├─ tests/
├─ .github/workflows/python-ci.yml
├─ requirements.txt
├─ LICENSE
└─ CITATION.cff
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train unsupervised (embeddings + AE reconstruction)
python src/train.py --config configs/advanced.yaml

# Evaluate & save anomaly scores + calibrated threshold
python src/evaluate.py --config configs/advanced.yaml

# Explain: SHAP on fused features
python -m src.explain.shap_explain --config configs/advanced.yaml --n_samples 256

# (Optional) Run the dashboard (needs artifacts/ outputs)
streamlit run src/dashboard_app.py -- --config configs/advanced.yaml
```

## Citation
```
@software{log_ad_microservices_advanced,
  title={Unsupervised Anomaly Detection for Log Data in Microservice Architectures — Advanced},
  year={2025},
  author={Your Name}
}
```
