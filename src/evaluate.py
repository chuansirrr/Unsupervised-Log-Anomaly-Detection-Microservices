
import yaml, torch, numpy as np, os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data.dataset import LogDataset
from src.data.parser import freq_vector
from src.models.transformer import TransformerEncoder
from src.models.lstm import LSTMEncoder
from src.models.fusion import AttentionFusion
from src.models.autoencoder import AutoEncoder
from src.models.ensemble import HybridAnomalyEnsemble

def evaluate(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    device = cfg['train']['device'] if torch.cuda.is_available() else 'cpu'

    ds = LogDataset(cfg['data']['log_dir'], cfg['model']['transformer_name'], cfg['data']['max_seq_len'], split='val', train_ratio=cfg['data']['train_ratio'])
    loader = DataLoader(ds, batch_size=cfg['data']['batch_size'], shuffle=False)

    ckpt = torch.load("artifacts/model.pt", map_location=device)
    vocab = ckpt['vocab']
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['transformer_name'])

    trans = TransformerEncoder(cfg['model']['transformer_name'], out_dim=cfg['model']['transformer_dim']).to(device); trans.load_state_dict(ckpt['trans']); trans.eval()
    lstm = LSTMEncoder(input_dim=tokenizer.vocab_size, hidden_dim=cfg['model']['lstm_hidden']).to(device); lstm.load_state_dict(ckpt['lstm']); lstm.eval()

    first = next(iter(loader))
    ids0 = first['input_ids'].to(device); mask0 = first['attention_mask'].to(device)
    with torch.no_grad():
        tdim = trans(ids0, mask0).shape[1]
        ldim = lstm(ids0).shape[1]
    fdim = tdim + ldim + len(vocab)
    fusion = AttentionFusion([tdim, ldim, len(vocab)], out_dim=fdim).to(device); fusion.load_state_dict(ckpt['fusion']); fusion.eval()
    ae = AutoEncoder(fdim, cfg['model']['ae_hidden']).to(device); ae.load_state_dict(ckpt['ae']); ae.eval()

    features = []; recon_errs = []; raws = []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
            tvec = trans(ids, mask)
            lvec = lstm(ids)
            fmat = np.stack([freq_vector(t, vocab, len(vocab)) for t in batch['template']], axis=0)
            fvec = torch.tensor(fmat, dtype=torch.float32, device=device)
            fused = fusion([tvec, lvec, fvec])
            recon, z = ae(fused)
            err = ((recon - fused)**2).mean(dim=1).cpu().numpy()
            features.append(z.cpu().numpy())
            recon_errs.append(err)
            raws.extend(batch['raw'])

    features = np.concatenate(features, axis=0)
    recon_errs = np.concatenate(recon_errs, axis=0)

    ens = HybridAnomalyEnsemble(gmm_components=cfg['evaluation']['gmm_components'])
    ens.fit(recon_errs, features)
    scores = ens.score(recon_errs, features)

    os.makedirs("artifacts", exist_ok=True)
    np.save("artifacts/features.npy", features)
    np.save("artifacts/recon_errors.npy", recon_errs)
    np.save("artifacts/scores.npy", scores)
    with open("artifacts/top_anomalies.txt", "w") as f:
        idx = scores.argsort()[::-1]
        for i in idx[:200]:
            f.write(f"{scores[i]:.4f}\t{raws[i]}\n")
    print("Saved artifacts: features.npy, recon_errors.npy, scores.npy, top_anomalies.txt")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    evaluate(args.config)
