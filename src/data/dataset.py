
import os, torch
from torch.utils.data import Dataset
from typing import List
from transformers import AutoTokenizer
from .parser import template_of

class LogDataset(Dataset):
    def __init__(self, log_dir: str, transformer_name: str, max_len: int = 256, split: str = "train", train_ratio: float = 0.8):
        files = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')])
        lines: List[str] = []
        for f in files:
            with open(f, 'r', errors='ignore') as fh:
                lines.extend([ln.strip() for ln in fh if ln.strip()])

        n = len(lines)
        cut = int(n * train_ratio)
        if split == "train":
            self.lines = lines[:cut]
        elif split in ("val", "test"):
            self.lines = lines[cut:]
        else:
            self.lines = lines

        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.max_len = max_len
        self.templates = []
        for ln in self.lines:
            templ, raw = template_of(ln)
            self.templates.append(templ)

    def __len__(self): return len(self.lines)

    def __getitem__(self, idx: int):
        text = self.lines[idx]
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "raw": text,
            "template": self.templates[idx]
        }
