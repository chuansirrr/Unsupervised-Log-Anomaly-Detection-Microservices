
import pandas as pd
import numpy as np

def build_service_graph(csv_path: str):
    df = pd.read_csv(csv_path)
    nodes = sorted(set(df["src_service"]).union(set(df["dst_service"])))
    idx = {n:i for i,n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=float)
    for _, r in df.iterrows():
        i = idx[r["src_service"]]; j = idx[r["dst_service"]]
        A[i,j] += float(r.get("weight", 1.0))
    row_sums = A.sum(axis=1, keepdims=True) + 1e-9
    A = A / row_sums
    return A, idx, nodes
