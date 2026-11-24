"""
fl_sim.py - FedAvg on FEMNIST in plain PyTorch. Each writer is one client.
Client selection is a plug-in function, so the EcoOrchestrator can replace the random baseline later.

Quick check (a few seconds):  python fl_sim.py --smoke
Baseline run:                 python fl_sim.py
"""
import argparse, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

from fl_data import load_femnist_clients


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 5, padding=2)
        self.c2 = nn.Conv2d(16, 32, 5, padding=2)
        self.f1 = nn.Linear(32 * 7 * 7, 128)
        self.f2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)      # 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.c2(x)), 2)      # 14x14 -> 7x7
        return self.f2(F.relu(self.f1(x.flatten(1))))


def to_tensor(X):
    return torch.from_numpy(X).float().div(255.0).unsqueeze(1)      # (N, 1, 28, 28), values 0..1


def local_train(model, global_state, X, y, epochs, lr, batch):
    """One client's local training, starting from the global weights. Returns (new weights, mean loss, all_finite)."""
    model.load_state_dict(global_state)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n, total, count = len(y), 0.0, 0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(X[b]), y[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)     # stops rare exploding updates
            opt.step()
            total += loss.item() * len(b)
            count += len(b)
    new = {k: v.detach().clone() for k, v in model.state_dict().items()}
    finite = np.isfinite(total) and all(bool(torch.isfinite(v).all()) for v in new.values())
    return new, total / max(count, 1), finite


def fedavg(states, sizes):
    """Average the clients' weights, weighted by how many training samples each has."""
    tot = float(sum(sizes))
    return {k: sum(s[k].float() * (n / tot) for s, n in zip(states, sizes)) for k in states[0]}


@torch.no_grad()
def evaluate(model, state, X, y):
    model.load_state_dict(state)
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(y), 1024):
        out = model(X[i:i + 1024])
        loss += F.cross_entropy(out, y[i:i + 1024], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + 1024]).sum().item()
    return correct / len(y), loss / len(y)


# ---------------- client selection (the plug-in point) ----------------
def select_random(ctx):
    """Baseline: k clients chosen uniformly at random.
    ctx has: round, num_clients, k, rng, last_loss (per client, NaN until first selected),
             times_selected (per client), clients (list of dicts with X, y, id, writer)."""
    return ctx["rng"].choice(ctx["num_clients"], size=ctx["k"], replace=False)


def run(ds, args, selector=select_random, tag="random"):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    clients, Xte, yte, num_classes = load_femnist_clients(
        ds, num_clients=args.clients, min_samples=args.min_samples, seed=args.seed)
    for c in clients:
        c["Xt"], c["yt"] = to_tensor(c["X"]), torch.from_numpy(c["y"])
    Xte_t, yte_t = to_tensor(Xte), torch.from_numpy(yte)
    n = len(clients)
    k = min(args.per_round, n)
    print(f"{n} clients | {sum(len(c['y']) for c in clients)} train samples | {len(yte)} test samples | "
          f"{num_classes} classes | {k} clients per round")

    model = Net(num_classes)
    state = {key: v.clone() for key, v in model.state_dict().items()}
    last_loss = np.full(n, np.nan)
    times_selected = np.zeros(n, dtype=int)
    log, t0 = [], time.time()

    for r in range(1, args.rounds + 1):
        ctx = {"round": r, "num_clients": n, "k": k, "rng": rng, "last_loss": last_loss,
               "times_selected": times_selected, "clients": clients}
        chosen = [int(i) for i in selector(ctx)]
        states, sizes, losses, skipped = [], [], [], 0
        for i in chosen:
            s, l, finite = local_train(model, state, clients[i]["Xt"], clients[i]["yt"], args.epochs, args.lr, args.batch)
            times_selected[i] += 1
            if not finite:                       # a diverged client update is dropped, never averaged in
                skipped += 1
                continue
            states.append(s); sizes.append(len(clients[i]["yt"])); losses.append(l)
            last_loss[i] = l
        if states:
            state = fedavg(states, sizes)
        acc, tloss = evaluate(model, state, Xte_t, yte_t)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        log.append({"round": r, "test_acc": acc, "test_loss": tloss, "mean_train_loss": mean_loss,
                    "skipped": skipped, "selected": " ".join(map(str, chosen))})
        if r == 1 or r % 5 == 0 or r == args.rounds:
            print(f"round {r:3d}/{args.rounds} | test acc {acc:.3f} | test loss {tloss:.3f} | "
                  f"train loss {mean_loss:.3f} | {time.time() - t0:.0f}s")
    df = pd.DataFrame(log)
    df.to_csv(f"fl_log_{tag}.csv", index=False)
    print(f"saved fl_log_{tag}.csv | dropped diverged client updates: {int(df['skipped'].sum())}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=100)
    p.add_argument("--per-round", type=int, default=10)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--min-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny run to check that everything works")
    a = p.parse_args()
    if a.smoke:
        a.clients, a.per_round, a.rounds, a.epochs = 8, 3, 2, 1
    from datasets import load_dataset
    data = load_dataset("flwrlabs/femnist", split="train")
    run(data, a)