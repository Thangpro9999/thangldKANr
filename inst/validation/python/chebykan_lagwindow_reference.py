"""ChebyKAN applied to lag windows, on the shared thangldKANr fixture.

Usage: python chebykan_lagwindow_reference.py <ChebyKAN clone> <fixture_dir> <out_dir>

ChebyKAN: https://github.com/SynodicMonth/ChebyKAN, commit
5f7efdd18e749bcc99481bd87dc90bdeafb920d8 (no license file; not copied).
Both layer versions are imported unchanged ("opt": cos(k acos z), "unopt":
recurrence). For every run this script independently (in NumPy):
  1. builds the lag windows x_t = (y_{t-l})_l, target y_t, from the raw series;
  2. holds out the last n_val windows (time order) for validation;
  3. standardizes with mean and sd (ddof = 1) of the training targets only;
then trains a plain stack of ChebyKANLayer with torch.optim.Adam on the
fixture's batch schedule, restores the best validation epoch and produces
the one-step forecast after the last observation. CPU, float64, one thread.
"""
import csv, importlib.util, json, os, platform, subprocess, sys, time
import numpy as np
import torch

PINNED = "5f7efdd18e749bcc99481bd87dc90bdeafb920d8"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([repr(float(v)) if isinstance(v, (float, np.floating)) else v for v in r])


def load_layer(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ChebyKANLayer


def read_params(prefix):
    dims = {int(r["layer"]): tuple(int(v) for v in r["dims"].split())
            for r in read_csv(prefix + "_dims.csv")}
    vals = {l: np.zeros(int(np.prod(d))) for l, d in dims.items()}
    for r in read_csv(prefix + "_params.csv"):
        vals[int(r["layer"])][int(r["pos"])] = float(r["value"])
    return [vals[l].reshape(dims[l], order="F") for l in sorted(dims)]


def write_params(path, arrs):
    rows = []
    for l, a in enumerate(arrs):
        flat = np.asarray(a).reshape(-1, order="F")
        rows += [[l, "coef", i, float(v)] for i, v in enumerate(flat)]
    write_rows(path, ["layer", "name", "pos", "value"], rows)


def main():
    clone, fix, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    head = subprocess.check_output(["git", "-C", clone, "rev-parse", "HEAD"], text=True).strip()
    variants = {"opt": load_layer(os.path.join(clone, "ChebyKANLayer.py"), "o"),
                "unopt": load_layer(os.path.join(clone, "ChebyKANLayer_unoptimized.py"), "u")}
    env = {"reference": "ChebyKAN on lag windows", "commit": head, "pinned_commit": PINNED,
           "commit_ok": head == PINNED, "python": sys.version.split()[0],
           "torch": torch.__version__, "numpy": np.__version__,
           "platform": platform.platform(), "threads": torch.get_num_threads()}
    json.dump(env, open(os.path.join(out, "python_env.json"), "w"), indent=2)
    st = read_csv(os.path.join(fix, "train_settings.csv"))[0]
    n_val = int(st["n_val"])
    timing = []
    for run in read_csv(os.path.join(fix, "train_runs.csv")):
        d = os.path.join(fix, "train", run["id"])
        y = np.array([float(r["y"]) for r in read_csv(os.path.join(d, "series.csv"))])
        meta = read_csv(os.path.join(d, "meta.csv"))[0]
        lags = [int(v) for v in meta["lags"].split()]
        idx = np.arange(max(lags), len(y))                     # 0-based targets
        X = np.stack([y[idx - l] for l in lags], axis=1)
        T = y[idx]
        ntr = len(T) - n_val
        center, scale = T[:ntr].mean(), T[:ntr].std(ddof=1)
        Xs, Ts = (X - center) / scale, (T - center) / scale
        x_last = (np.array([y[len(y) - l] for l in lags]) - center) / scale
        od = os.path.join(out, "train", run["id"])
        os.makedirs(od, exist_ok=True)
        write_rows(os.path.join(od, "windows.csv"), ["row"] + [f"lag{l}" for l in lags] + ["target"],
                   [[i] + list(X[i]) + [T[i]] for i in range(len(T))])
        write_rows(os.path.join(od, "scaling.csv"), ["center", "scale"], [[center, scale]])
        init = read_params(os.path.join(d, "init"))
        sched = {}
        for r in read_csv(os.path.join(d, "schedule.csv")):
            sched.setdefault(int(r["epoch"]), {}).setdefault(int(r["batch"]), []).append(int(r["idx"]))
        sched = [[sched[e][b] for b in sorted(sched[e])] for e in sorted(sched)]
        xtr, ytr = torch.tensor(Xs[:ntr]), torch.tensor(Ts[:ntr]).reshape(-1, 1)
        xva, yva = torch.tensor(Xs[ntr:]), torch.tensor(Ts[ntr:]).reshape(-1, 1)
        for v, Layer in variants.items():
            layers = []
            for a in init:
                lay = Layer(a.shape[0], a.shape[1], a.shape[2] - 1)
                lay.double()
                with torch.no_grad():
                    lay.cheby_coeffs.copy_(torch.tensor(a))
                layers.append(lay)
            net = torch.nn.Sequential(*layers)
            opt = torch.optim.Adam(net.parameters(), lr=float(st["lr"]),
                                   betas=(float(st["beta1"]), float(st["beta2"])),
                                   eps=float(st["eps"]))
            mse = torch.nn.functional.mse_loss
            best, best_epoch, best_state, hist = float("inf"), None, None, []
            t0 = time.perf_counter()
            for e, batches in enumerate(sched):
                for b in batches:
                    opt.zero_grad()
                    loss = mse(net(xtr[b]), ytr[b])
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    trl, val = mse(net(xtr), ytr).item(), mse(net(xva), yva).item()
                hist.append([e + 1, trl, val])
                if val < best:
                    best, best_epoch = val, e + 1
                    best_state = [l.cheby_coeffs.detach().clone().numpy() for l in layers]
            fit_sec = time.perf_counter() - t0
            with torch.no_grad():
                for l, a in zip(layers, best_state):
                    l.cheby_coeffs.copy_(torch.tensor(a))
                fc = net(torch.tensor(x_last).reshape(1, -1)).item() * scale + center
            write_rows(os.path.join(od, f"{v}_history.csv"), ["epoch", "train_loss", "val_loss"], hist)
            write_params(os.path.join(od, f"{v}_best_params.csv"), best_state)
            write_rows(os.path.join(od, f"{v}_forecast.csv"), ["forecast"], [[fc]])
            timing.append([run["id"], v, best_epoch, fit_sec])
    write_rows(os.path.join(out, "train_timing.csv"), ["id", "variant", "best_epoch", "fit_sec"], timing)
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
