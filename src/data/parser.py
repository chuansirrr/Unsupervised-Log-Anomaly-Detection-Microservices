
import re
from collections import Counter
from typing import List, Dict, Tuple

TEMPLATE_TOKEN = "<*>"
DIGIT_RE = re.compile(r"\d+")
HEX_RE = re.compile(r"0x[0-9a-fA-F]+")

def normalize_token(tok: str) -> str:
    t = HEX_RE.sub(TEMPLATE_TOKEN, tok)
    t = DIGIT_RE.sub(TEMPLATE_TOKEN, t)
    return t

def tokenize_line(line: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_:/\.-]+", line)
    return toks

def template_of(line: str) -> Tuple[str, List[str]]:
    toks = tokenize_line(line)
    templ_toks = [normalize_token(t) for t in toks]
    template = " ".join(templ_toks)
    return template, toks

def topk_vocab(templates: List[str], k: int = 512) -> Dict[str, int]:
    cnt = Counter()
    for t in templates:
        cnt.update(t.split())
    common = cnt.most_common(k)
    return {tok:i for i,(tok,_) in enumerate(common)}

def freq_vector(template: str, vocab: Dict[str,int], k:int) -> list:
    vec = [0]*k
    for tok in template.split():
        if tok in vocab:
            vec[vocab[tok]] += 1
    return vec
