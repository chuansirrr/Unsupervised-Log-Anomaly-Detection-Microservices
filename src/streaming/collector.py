
import time, os, random
from datetime import datetime

TEMPLATES = [
    "GET /api/v1/users/{id} 200 OK svc=user",
    "POST /api/v1/login 401 FAIL svc=auth",
    "PUT /api/v1/orders/{id} 500 ERR svc=orders",
    "GET /api/v1/inventory 200 OK svc=inv",
    "RPC call payment->orders timeout=1500ms",
    "RPC call orders->ship latency=300ms"
]

ANOMALIES = [
    "POST /api/v1/payment 502 BAD_GATEWAY svc=payment",
    "RPC call payment->orders timeout=9000ms",
    "GET /api/v1/users/{id} 429 THROTTLED svc=user"
]

def stream_to_dir(out_dir: str, rate_per_sec: int = 50, anomaly_rate: float = 0.05):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, "stream.log")
    with open(fname, "a") as f:
        while True:
            is_anom = random.random() < anomaly_rate
            line = random.choice(ANOMALIES if is_anom else TEMPLATES)
            line = line.replace("{id}", str(random.randint(1, 100000)))
            f.write(f"{datetime.utcnow().isoformat()}Z {line}\n")
            f.flush()
            time.sleep(1.0 / max(1, rate_per_sec))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/sample_logs")
    ap.add_argument("--rate", type=int, default=50)
    ap.add_argument("--anomaly_rate", type=float, default=0.05)
    args = ap.parse_args()
    stream_to_dir(args.out_dir, args.rate, args.anomaly_rate)
