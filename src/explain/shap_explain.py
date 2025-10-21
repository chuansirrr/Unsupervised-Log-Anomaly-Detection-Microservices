
import yaml, numpy as np, shap, torch
from transformers import AutoTokenizer
from ..data.dataset import LogDataset
from ..models.transformer import TransformerEncoder
from ..models.autoencoder import AutoEncoder
from ..models.lstm import LSTMEncoder
from ..models.fusion import AttentionFusion
from ..data.parser import topk_vocab, freq_vector

def main(cfg_path, n_samples=256):
    cfg = yaml.safe_load(open(cfg_path))
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['transformer_name'])
    ds = LogDataset(cfg['data']['log_dir'], cfg['model']['transformer_name'], cfg['data']['max_seq_len'], split='val', train_ratio=cfg['data']['train_ratio'])
    X = [ds[i] for i in range(min(len(ds), n_samples))]

    templates = [x['template'] for x in X]
    vocab = topk_vocab(templates, k=cfg['model']['freq_topk'])

    trans = TransformerEncoder(cfg['model']['transformer_name'], out_dim=cfg['model']['transformer_dim'])
    lstm = LSTMEncoder(input_dim=tokenizer.vocab_size, hidden_dim=cfg['model']['lstm_hidden'])
    fusion = AttentionFusion([cfg['model']['transformer_dim'], cfg['model']['lstm_hidden'], len(vocab)], out_dim=256)
    ae = AutoEncoder(256, cfg['model']['ae_hidden'])

    def features(batch):
        with torch.no_grad():
            ids = torch.stack([b['input_ids'] for b in batch])
            mask = torch.stack([b['attention_mask'] for b in batch])
            tvec = trans(ids, mask)
            lvec = lstm(ids)
            fmat = np.stack([freq_vector(b['template'], vocab, len(vocab)) for b in batch], axis=0)
            fvec = torch.tensor(fmat, dtype=torch.float32)
            fused = fusion([tvec, lvec, fvec])
        return fused.numpy()

    def score_fn(F):
        F = torch.tensor(F, dtype=torch.float32)
        with torch.no_grad():
            recon, _ = ae(F)
        err = ((recon - F)**2).mean(dim=1).numpy()
        return err

    B = features(X[:min(32, len(X))])
    explainer = shap.KernelExplainer(score_fn, B)
    F = features(X)
    shap_values = explainer.shap_values(F, nsamples=100)
    np.save("artifacts/shap_values.npy", shap_values, allow_pickle=True)
    print("Saved artifacts/shap_values.npy")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_samples", type=int, default=256)
    args = ap.parse_args()
    main(args.config, args.n_samples)
