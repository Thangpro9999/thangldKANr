"""Graph KAN reference: NumPy normalized adjacency + ChebyKAN layers + torch Adam.

Usage: python graphkan_reference.py <ChebyKAN clone> <fixture_dir> <out_dir>

Independent of thangldKANr except for the fixture (edge lists, features,
targets, node masks, initial coefficients). This script
  * builds A_hat = D^-1/2 (A + I) D^-1/2 from the edge list with NumPy,
  * standardizes features and target with training-node mean and sd (ddof 1),
  * stacks ChebyKANLayer (https://github.com/SynodicMonth/ChebyKAN, commit
    5f7efdd18e749bcc99481bd87dc90bdeafb920d8; no license file, nothing copied;
    "opt" = acos version, "unopt" = recurrence version) as
    H_l = ChebyKANLayer_l(A_hat @ H_{l-1}), or, for cases with self_branch = 1
    in meta.csv, H_l = ChebyKANLayer_l(cat([H_{l-1}, A_hat @ H_{l-1}], dim=1)),
  * trains full batch with torch.optim.Adam on the MSE over training nodes,
    recording train/validation loss after every step and restoring the epoch
    with the lowest validation loss (same early-stopping rule as thangldKANr).
"""
import csv, importlib.util, json, os, platform, subprocess, sys
import numpy as np
import torch

PINNED = "5f7efdd18e749bcc99481bd87dc90bdeafb920d8"


def rd(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_case(d):
    meta = rd(os.path.join(d, "meta.csv"))[0]
    widths = [int(v) for v in meta["widths"].split()]
    n = int(meta["n"])
    dims = {int(r["layer"]): tuple(int(v) for v in r["dims"].split())
            for r in rd(os.path.join(d, "dims.csv"))}
    vals = {l: np.zeros(int(np.prod(s))) for l, s in dims.items()}
    for r in rd(os.path.join(d, "params.csv")):
        vals[int(r["layer"])][int(r["pos"])] = float(r["value"])
    coefs = [vals[l].reshape(dims[l], order="F") for l in sorted(dims)]
    edges = [(int(r["from"]), int(r["to"])) for r in rd(os.path.join(d, "edges.csv"))]
    xr = rd(os.path.join(d, "x.csv"))
    x = np.zeros((n, widths[0]))
    for r in xr:
        x[int(r["row"]), int(r["col"])] = float(r["value"])
    y = np.array([float(r["y"]) for r in rd(os.path.join(d, "y.csv"))])
    nodes = rd(os.path.join(d, "nodes.csv"))
    a = sorted(int(r["node"]) for r in nodes if r["set"] == "a")
    b = sorted(int(r["node"]) for r in nodes if r["set"] == "b")
    sb = int(meta.get("self_branch", "0") or 0) == 1
    return widths, n, coefs, edges, x, y, a, b, sb


def a_hat(edges, n):
    A = np.zeros((n, n))
    for i, j in edges:
        if i != j:
            A[i, j] = 1.0
            A[j, i] = 1.0
    A = A + np.eye(n)
    d = A.sum(axis=1)
    dinv = 1.0 / np.sqrt(d)
    return dinv[:, None] * A * dinv[None, :]


class Net(torch.nn.Module):
    def __init__(self, Layer, widths, coefs, self_branch=False):
        super().__init__()
        self.self_branch = self_branch
        self.layers = torch.nn.ModuleList()
        for (p, q), c in zip(zip(widths[:-1], widths[1:]), coefs):
            lay = Layer(p * (2 if self_branch else 1), q, c.shape[2] - 1)
            lay.double()
            with torch.no_grad():
                lay.cheby_coeffs.copy_(torch.tensor(c))
            self.layers.append(lay)

    def forward_layers(self, x, A):
        outs = []
        for lay in self.layers:
            u = torch.cat([x, A @ x], dim=1) if self.self_branch else A @ x
            x = lay(u)
            outs.append(x)
        return outs

    def forward(self, x, A):
        return self.forward_layers(x, A)[-1]

    def coefs(self, kind="params", opt=None):
        out = []
        for lay in self.layers:
            t = lay.cheby_coeffs
            out.append((t.detach() if kind == "params" else
                        t.grad.detach() if kind == "grad" else
                        opt.state[t]["exp_avg_sq"]).numpy().copy())
        return out


def write_rows(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([repr(float(v)) if isinstance(v, (float, np.floating)) else v for v in r])


def flat(arrs):
    return np.concatenate([np.asarray(a).reshape(-1, order="F") for a in arrs])


def write_vec(path, v, name="value"):
    write_rows(path, [name], [[float(z)] for z in np.asarray(v).reshape(-1, order="F")])


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ChebyKANLayer


def net_case(variants, case_dir, od):
    mse = torch.nn.functional.mse_loss
    widths, n, coefs, edges, x, y, nodes, _, sb = load_case(case_dir)
    A = a_hat(edges, n)
    write_vec(os.path.join(od, "net_a_hat.csv"), A)
    for v, Layer in variants.items():
        net = Net(Layer, widths, coefs, sb)
        At = torch.tensor(A)
        xt = torch.tensor(x, requires_grad=True)
        idx = torch.tensor(nodes)
        yt = torch.tensor(y[nodes]).reshape(-1, 1)
        outs = net.forward_layers(xt, At)
        for l, o in enumerate(outs):
            write_vec(os.path.join(od, "net_%s_layer%d.csv" % (v, l)), o.detach().numpy())
        loss = mse(outs[-1][idx], yt)
        loss.backward()
        write_vec(os.path.join(od, "net_%s_grad_params.csv" % v), flat(net.coefs("grad")))
        write_vec(os.path.join(od, "net_%s_grad_x.csv" % v), xt.grad.numpy())
        write_rows(os.path.join(od, "net_%s_loss.csv" % v), ["loss"], [[loss.item()]])
        net = Net(Layer, widths, coefs, sb)
        opt = torch.optim.Adam(net.parameters(), lr=0.02, betas=(0.9, 0.999), eps=1e-8)
        rows = []
        xt = torch.tensor(x)
        for step in range(5):
            opt.zero_grad()
            loss = mse(net(xt, At)[idx], yt)
            loss.backward()
            opt.step()
            rows += [[step, i, float(val)] for i, val in enumerate(flat(net.coefs()))]
        write_rows(os.path.join(od, "adam_%s_params.csv" % v), ["step", "pos", "value"], rows)



def main(clone, fix, out):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    os.makedirs(out, exist_ok=True)
    head = subprocess.check_output(["git", "-C", clone, "rev-parse", "HEAD"], text=True).strip()
    variants = {"opt": load(os.path.join(clone, "ChebyKANLayer.py"), "ck_opt"),
                "unopt": load(os.path.join(clone, "ChebyKANLayer_unoptimized.py"), "ck_unopt")}
    env = {"reference": "NumPy A_hat + ChebyKANLayer stack + torch.optim.Adam (full batch)",
           "url": "https://github.com/SynodicMonth/ChebyKAN", "commit": head,
           "pinned_commit": PINNED, "commit_ok": head == PINNED,
           "python": sys.version.split()[0], "torch": torch.__version__,
           "numpy": np.__version__, "platform": platform.platform(), "threads": 1,
           "dtype": "float64"}
    json.dump(env, open(os.path.join(out, "python_env.json"), "w"), indent=2)
    mse = torch.nn.functional.mse_loss

    # 1. forward, gradient, 5 Adam steps on the net fixtures (features as given);
    #    "net" -> out/, "net_self" (self branch) -> out/self/
    for case, od in (("net", out), ("net_self", os.path.join(out, "self"))):
        if not os.path.isdir(os.path.join(fix, case)):
            continue
        os.makedirs(od, exist_ok=True)
        net_case(variants, os.path.join(fix, case), od)
    # 2. training runs
    st = rd(os.path.join(fix, "train_settings.csv"))[0]
    epochs, lr, pat = int(st["epochs"]), float(st["lr"]), int(st["patience"])
    betas, eps = (float(st["beta1"]), float(st["beta2"])), float(st["eps"])
    timing = []
    for run in rd(os.path.join(fix, "train_runs.csv")):
        rid = run["id"]
        widths, n, coefs, edges, x, y, tr, va, sb = load_case(os.path.join(fix, "train", rid))
        d = os.path.join(out, "train", rid)
        os.makedirs(d, exist_ok=True)
        xc, xs_ = x[tr].mean(axis=0), x[tr].std(axis=0, ddof=1)
        yc, ys_ = y[tr].mean(), y[tr].std(ddof=1)
        xs = (x - xc) / xs_
        yz = (y - yc) / ys_
        write_rows(os.path.join(d, "scaling.csv"), ["what", "value"],
                   [["x_center", float(v)] for v in xc] + [["x_scale", float(v)] for v in xs_] +
                   [["y_center", yc], ["y_scale", ys_]])
        A = torch.tensor(a_hat(edges, n))
        xt = torch.tensor(xs)
        itr, iva = torch.tensor(tr), torch.tensor(va)
        ytr = torch.tensor(yz[tr]).reshape(-1, 1)
        for v, Layer in variants.items():
            net = Net(Layer, widths, coefs, sb)
            opt = torch.optim.Adam(net.parameters(), lr=lr, betas=betas, eps=eps)
            hist, best, best_ep, best_c, wait = [], np.inf, None, None, 0
            import time
            t0 = time.perf_counter()
            for ep in range(1, epochs + 1):
                opt.zero_grad()
                loss = mse(net(xt, A)[itr], ytr)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    o = net(xt, A).numpy().reshape(-1)
                trl = float(np.mean((o[tr] - yz[tr]) ** 2))
                vl = float(np.mean((o[va] - yz[va]) ** 2))
                hist.append([ep, trl, vl])
                if vl < best or best_ep is None:
                    best, best_ep, best_c, wait = vl, ep, net.coefs(), 0
                else:
                    wait += 1
                if wait >= pat:
                    break
            timing.append([rid, v, best_ep, time.perf_counter() - t0])
            write_rows(os.path.join(d, "%s_history.csv" % v), ["epoch", "train_loss", "val_loss"],
                       hist)
            write_vec(os.path.join(d, "%s_best_params.csv" % v), flat(best_c))
            nb = Net(Layer, widths, best_c, sb)
            with torch.no_grad():
                pred = nb(xt, A).numpy().reshape(-1) * ys_ + yc
            write_vec(os.path.join(d, "%s_predictions.csv" % v), pred, "pred")
    write_rows(os.path.join(out, "train_timing.csv"), ["id", "variant", "best_epoch", "fit_sec"],
               timing)


if __name__ == "__main__":
    main(*sys.argv[1:4])
