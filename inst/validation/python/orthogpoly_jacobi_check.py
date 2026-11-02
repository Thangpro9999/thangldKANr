"""Side check: do two public JacobiKAN layers reproduce Legendre at a = b = 0?

Usage: python orthogpoly_jacobi_check.py <JacobiKAN clone> <OrthogPolyKANs clone>

Jacobi polynomials with a = b = 0 are the Legendre polynomials. This script
evaluates P_0..P_4 through each layer (one input, one output, unit
coefficient on one degree) and prints the maximum deviation from NumPy's
Legendre polynomials. Used to choose the numerical reference of thangldKANr (basis = "jacobi").
"""
import importlib.util, os, sys
import numpy as np
import torch
from numpy.polynomial import legendre

torch.set_default_dtype(torch.float64)


def load(path):
    s = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.JacobiKANLayer


z = np.array([-0.9, -0.3, 0.2, 0.7])
x = torch.tensor(np.arctanh(z)).reshape(-1, 1)
leg = np.stack([legendre.legval(z, [0] * k + [1]) for k in range(5)], 1)
for name, path in [("SpaceLearner/JacobiKAN", os.path.join(sys.argv[1], "JacobiKANLayer.py")),
                   ("OrthogPolyKANs/JacobiKAN",
                    os.path.join(sys.argv[2], "JacobiKAN", "JacobiKANLayer.py"))]:
    lay = load(path)(1, 1, 4, 0.0, 0.0)
    err = []
    for k in range(5):
        with torch.no_grad():
            lay.jacobi_coeffs.zero_()
            lay.jacobi_coeffs[0, 0, k] = 1
            err.append(float(np.max(np.abs(lay(x).numpy().ravel() - leg[:, k]))))
    print(name, "max |P_k - Legendre_k|, k = 0..4:", ["%.3g" % e for e in err])
