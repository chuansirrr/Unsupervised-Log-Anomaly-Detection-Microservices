
import torch, yaml, numpy as np
from ..models.transformer import TransformerEncoder
from ..data.dataset import LogDataset

def main(cfg_path, n_samples=64):
    cfg = yaml.safe_load(open(cfg_path))
    ds = LogDataset(cfg['data']['log_dir'], cfg['model']['transformer_name'], cfg['data']['max_seq_len'], split='val', train_ratio=cfg['data']['train_ratio'])
    enc = TransformerEncoder(cfg['model']['transformer_name'], out_dim=cfg['model']['transformer_dim'])

    batch = [ds[i] for i in range(min(n_samples, len(ds)))]
    ids = torch.stack([b['input_ids'] for b in batch]); mask = torch.stack([b['attention_mask'] for b in batch])
    ids = ids.clone().detach().requires_grad_(True)
    out = enc(ids, mask)
    obj = (out**2).sum()
    obj.backward()
    sal = ids.grad.abs().detach().cpu().numpy()
    np.save("artifacts/token_saliency.npy", sal)
    print("Saved artifacts/token_saliency.npy")

if __name__ == "__main__":
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_samples", type=int, default=64)
    args = ap.parse_args()
    main(args.config, args.n_samples)
