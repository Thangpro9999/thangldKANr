"""Run the safe Pade function of rational_activations on the thangldKANr (basis = "rational") fixture.

Usage: python rationalkan_reference.py <rational_activations clone> <fixture_dir> <out_dir>

Clone https://github.com/ml-research/rational_activations at commit
2496d918d924b117ef06ec35bdca7f207720c62c (MIT License). The pure PyTorch
function rational/torch/rational_pytorch_functions.py::Rational_PYTORCH_B_F
(P(x) / (1 + |b_1 x + ... + b_n x^n|)) is imported unchanged from that file
and applied to every edge with that edge's coefficients; the edge outputs
are summed over inputs (KAN layer). Nothing is copied into thangldKANr (basis = "rational").
"""
# ---- common driver (identical in every package of the ecosystem) -------------
import csv, json, os, platform, subprocess, sys, time
import numpy as np
import torch


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_matrix(path):
    rows = read_csv(path)
    nr = max(int(r["row"]) for r in rows) + 1
    nc = max(int(r["col"]) for r in rows) + 1
    m = np.zeros((nr, nc))
    for r in rows:
        m[int(r["row"]), int(r["col"])] = float(r["value"])
    return m


def read_vector(path, col="y"):
    return np.array([float(r[col]) for r in read_csv(path)])


def read_params(prefix):
    """R parameters -> list (per layer) of dicts name -> numpy array."""
    dims = {}
    for r in read_csv(prefix + "_dims.csv"):
        dims[(int(r["layer"]), r["name"])] = tuple(int(v) for v in r["dims"].split())
    vals = {k: np.zeros(int(np.prod(d))) for k, d in dims.items()}
    for r in read_csv(prefix + "_params.csv"):
        vals[(int(r["layer"]), r["name"])][int(r["pos"])] = float(r["value"])
    n_layers = max(k[0] for k in dims) + 1
    out = [dict() for _ in range(n_layers)]
    for (l, nm), v in vals.items():
        out[l][nm] = v.reshape(dims[(l, nm)], order="F")
    return out


def write_rows(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([repr(float(v)) if isinstance(v, (float, np.floating)) else v
                        for v in r])


def write_params(path, plist):
    rows = []
    for l, d in enumerate(plist):
        for nm, a in d.items():
            flat = np.asarray(a).reshape(-1, order="F")
            rows += [[l, nm, i, float(v)] for i, v in enumerate(flat)]
    write_rows(path, ["layer", "name", "pos", "value"], rows)


def write_matrix(path, m):
    m = np.atleast_2d(m)
    write_rows(path, ["row", "col", "value"],
               [[r, c, float(m[r, c])] for c in range(m.shape[1])
                for r in range(m.shape[0])])


def read_schedule(path):
    sched = {}
    for r in read_csv(path):
        sched.setdefault(int(r["epoch"]), {}).setdefault(int(r["batch"]), []).append(int(r["idx"]))
    return [[sched[e][b] for b in sorted(sched[e])] for e in sorted(sched)]


def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception as e:
        return "unknown (%s)" % e


def run(adapter, clone, fix, out):
    """adapter: dict with
       variants: {name: builder(widths, plist) -> (net, tensors)}
       tensors(net) -> list of dict name -> torch tensor (our parameterization)
       to_r(net, kind, opt=None) -> list of dict name -> numpy (kind in
            'params', 'grad', 'exp_avg_sq')"""
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    env = {"reference": adapter["name"], "url": adapter["url"],
           "commit": git_head(clone), "pinned_commit": adapter["commit"],
           "python": sys.version.split()[0], "torch": torch.__version__,
           "numpy": np.__version__, "platform": platform.platform(),
           "threads": torch.get_num_threads(), "dtype": "float64",
           "configuration": adapter["config_note"]}
    env["commit_ok"] = env["commit"] == env["pinned_commit"]
    with open(os.path.join(out, "python_env.json"), "w") as f:
        json.dump(env, f, indent=2)
    with open(os.path.join(out, "variants.txt"), "w") as f:
        f.write(" ".join(adapter["variants"]))
    mse = torch.nn.functional.mse_loss
    widths = [int(v) for v in read_csv(os.path.join(fix, "net_meta.csv"))[0]["widths"].split()]
    plist = read_params(os.path.join(fix, "net"))
    x = read_matrix(os.path.join(fix, "net_x.csv"))
    y = read_vector(os.path.join(fix, "net_y.csv"))
    for v, builder in adapter["variants"].items():
        net = builder(widths, plist)
        xt = torch.tensor(x, requires_grad=True)
        outs = net.forward_layers(xt)
        loss = mse(outs[-1], torch.tensor(y).reshape(-1, 1))
        loss.backward()
        for l, o in enumerate(outs):
            write_matrix(os.path.join(out, f"net_{v}_layer{l}_out.csv"),
                         o.detach().numpy())
        write_params(os.path.join(out, f"net_{v}_grad_params.csv"),
                     net.to_r("grad"))
        write_matrix(os.path.join(out, f"net_{v}_grad_x.csv"), xt.grad.numpy())
    st = read_csv(os.path.join(fix, "adam_settings.csv"))[0]
    sched = read_schedule(os.path.join(fix, "adam_schedule.csv"))
    for v, builder in adapter["variants"].items():
        net = builder(widths, plist)
        opt = torch.optim.Adam(net.parameters(), lr=float(st["lr"]),
                               betas=(float(st["beta1"]), float(st["beta2"])),
                               eps=float(st["eps"]))
        xt, yt = torch.tensor(x), torch.tensor(y).reshape(-1, 1)
        losses = []
        for s, batches in enumerate(sched):
            idx = batches[0]
            opt.zero_grad()
            loss = mse(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            losses.append([s, loss.item()])
            write_params(os.path.join(out, f"adam_{v}_step{s}_params.csv"),
                         net.to_r("params"))
            write_params(os.path.join(out, f"adam_{v}_step{s}_exp_avg_sq.csv"),
                         net.to_r("exp_avg_sq", opt))
        write_rows(os.path.join(out, f"adam_{v}_loss.csv"), ["step", "loss"], losses)
    ts = read_csv(os.path.join(fix, "train_settings.csv"))[0]
    timing = []
    for run_ in read_csv(os.path.join(fix, "train_runs.csv")):
        d = os.path.join(fix, "train", run_["id"])
        xtr = torch.tensor(read_matrix(os.path.join(d, "x_train.csv")))
        ytr = torch.tensor(read_vector(os.path.join(d, "y_train.csv"))).reshape(-1, 1)
        xva = torch.tensor(read_matrix(os.path.join(d, "x_val.csv")))
        yva = torch.tensor(read_vector(os.path.join(d, "y_val.csv"))).reshape(-1, 1)
        xte = torch.tensor(read_matrix(os.path.join(d, "x_test.csv")))
        w = [int(v) for v in read_csv(os.path.join(d, "init_meta.csv"))[0]["widths"].split()]
        init = read_params(os.path.join(d, "init"))
        sched = read_schedule(os.path.join(d, "schedule.csv"))
        od = os.path.join(out, "train", run_["id"])
        os.makedirs(od, exist_ok=True)
        for v, builder in adapter["variants"].items():
            net = builder(w, init)
            opt = torch.optim.Adam(net.parameters(), lr=float(ts["lr"]),
                                   betas=(float(ts["beta1"]), float(ts["beta2"])),
                                   eps=float(ts["eps"]))
            best, best_epoch, best_state, hist = float("inf"), None, None, []
            t0 = time.perf_counter()
            for e, batches in enumerate(sched):
                for idx in batches:
                    opt.zero_grad()
                    loss = mse(net(xtr[idx]), ytr[idx])
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    trl = mse(net(xtr), ytr).item()
                    val = mse(net(xva), yva).item()
                hist.append([e + 1, trl, val])
                if val < best:
                    best, best_epoch = val, e + 1
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            fit_sec = time.perf_counter() - t0
            net.load_state_dict(best_state)
            with torch.no_grad():
                t1 = time.perf_counter()
                pred = net(xte).numpy().ravel()
                pred_sec = time.perf_counter() - t1
            write_rows(os.path.join(od, f"{v}_history.csv"),
                       ["epoch", "train_loss", "val_loss"], hist)
            write_params(os.path.join(od, f"{v}_best_params.csv"), net.to_r("params"))
            write_rows(os.path.join(od, f"{v}_pred_test_scaled.csv"), ["pred"],
                       [[p] for p in pred])
            timing.append([run_["id"], v, best_epoch, fit_sec, pred_sec])
    write_rows(os.path.join(out, "train_timing.csv"),
               ["id", "variant", "best_epoch", "fit_sec", "predict_sec"], timing)
    print(json.dumps(env, indent=2))

# ---- rational_activations adapter ----------------------------------------------
import importlib.util


def load_fun(clone):
    spec = importlib.util.spec_from_file_location(
        "rational_ref", os.path.join(clone, "rational", "torch",
                                     "rational_pytorch_functions.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Rational_PYTORCH_B_F


class Net(torch.nn.Module):
    def __init__(self, fun, widths, plist):
        super().__init__()
        self.fun = fun
        self.num = torch.nn.ParameterList()
        self.den = torch.nn.ParameterList()
        for l in range(len(widths) - 1):
            self.num.append(torch.nn.Parameter(torch.tensor(plist[l]["num"])))
            self.den.append(torch.nn.Parameter(torch.tensor(plist[l]["den"])))

    def layer(self, l, x):
        num, den = self.num[l], self.den[l]
        p, q = num.shape[0], num.shape[1]
        cols = []
        for j in range(q):
            s = 0
            for i in range(p):
                s = s + self.fun(x[:, i].contiguous(), num[i, j], den[i, j], self.training)
            cols.append(s)
        return torch.stack(cols, dim=1)

    def forward_layers(self, x):
        outs = []
        for l in range(len(self.num)):
            x = self.layer(l, x)
            outs.append(x)
        return outs

    def forward(self, x):
        return self.forward_layers(x)[-1]

    def to_r(self, kind, opt=None):
        out = []
        for l in range(len(self.num)):
            d = {}
            for nm, t in (("num", self.num[l]), ("den", self.den[l])):
                d[nm] = (t.detach() if kind == "params" else
                         t.grad.detach() if kind == "grad" else
                         opt.state[t]["exp_avg_sq"]).numpy()
            out.append(d)
        return out


if __name__ == "__main__":
    clone, fix, out = sys.argv[1:4]
    fun = load_fun(clone)
    run({"name": "rational_activations Rational_PYTORCH_B_F (safe Pade, per edge)",
         "url": "https://github.com/ml-research/rational_activations",
         "commit": "2496d918d924b117ef06ec35bdca7f207720c62c",
         "config_note": "Rational_PYTORCH_B_F unchanged, one coefficient set per edge, summed over inputs",
         "variants": {"rational_b": lambda w, p: Net(fun, w, p)}},
        clone, fix, out)
