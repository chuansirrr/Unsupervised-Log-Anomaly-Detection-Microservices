
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from umap import UMAP

def umap_plot(features_path='artifacts/features.npy'):
    Z = np.load(features_path)
    emb = UMAP(n_components=2, random_state=42).fit_transform(Z)
    plt.figure()
    plt.scatter(emb[:,0], emb[:,1], s=8)
    plt.title("UMAP of Fused Embeddings")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    umap_plot()
