
import yaml, torch, os, numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data.dataset import LogDataset
from src.data.parser import topk_vocab, freq_vector
from src.models.transformer import TransformerEncoder
from src.models.lstm import LSTMEncoder
from src.models.fusion import AttentionFusion
from src.models.autoencoder import AutoEncoder

def train(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    device = cfg['train']['device'] if torch.cuda.is_available() else 'cpu'

    tr_ds = LogDataset(cfg['data']['log_dir'], cfg['model']['transformer_name'], cfg['data']['max_seq_len'], split='train', train_ratio=cfg['data']['train_ratio'])
    va_ds = LogDataset(cfg['data']['log_dir'], cfg['model']['transformer_name'], cfg['data']['max_seq_len'], split='val', train_ratio=cfg['data']['train_ratio'])
    tr_loader = DataLoader(tr_ds, batch_size=cfg['data']['batch_size'], shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=cfg['data']['batch_size'], shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['transformer_name'])
    vocab = topk_vocab(tr_ds.templates, k=cfg['model']['freq_topk'])

    trans = TransformerEncoder(cfg['model']['transformer_name'], out_dim=cfg['model']['transformer_dim']).to(device)
    lstm = LSTMEncoder(input_dim=tokenizer.vocab_size, hidden_dim=cfg['model']['lstm_hidden']).to(device)

    dummy_ids = torch.zeros((1, cfg['data']['max_seq_len']), dtype=torch.long).to(device)
    dummy_mask = torch.ones_like(dummy_ids).to(device)
    with torch.no_grad():
        tdim = trans(dummy_ids, dummy_mask).shape[1]
        ldim = lstm(dummy_ids).shape[1]
    fdim = tdim + ldim + len(vocab)
    fusion = AttentionFusion([tdim, ldim, len(vocab)], out_dim=fdim).to(device)

    ae = AutoEncoder(fdim, cfg['model']['ae_hidden']).to(device)
    opt = torch.optim.Adam(list(trans.parameters()) + list(lstm.parameters()) + list(fusion.parameters()) + list(ae.parameters()), lr=cfg['train']['lr'])

    for epoch in range(cfg['train']['epochs']):
        trans.train(); lstm.train(); fusion.train(); ae.train()
        total = 0.0
        for batch in tr_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            with torch.no_grad():
                tvec = trans(ids, mask)
            lvec = lstm(ids)
            fmat = np.stack([freq_vector(t, vocab, len(vocab)) for t in batch['template']], axis=0)
            fvec = torch.tensor(fmat, dtype=torch.float32, device=device)
            fused = fusion([tvec, lvec, fvec])
            recon, _ = ae(fused)
            loss = ((recon - fused)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} - recon MSE: {total/len(tr_loader):.4f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save({'trans': trans.state_dict(),
                'lstm': lstm.state_dict(),
                'fusion': fusion.state_dict(),
                'ae': ae.state_dict(),
                'vocab': vocab}, "artifacts/model.pt")
    print("Saved model to artifacts/model.pt")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)
