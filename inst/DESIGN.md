# thangldKANr design: the basis registry and how splineKANr plugs in

## Layers of the package

1. **Engine** (`R/engine.R`, `R/api.R`, `R/preprocess.R`, `R/validate.R`):
   forward/backward passes, central differences, Adam with both backends,
   early stopping, prediction, edge plots, save/load, validation. It knows
   nothing about any basis; it calls the generic `layer_*()` functions.
2. **Registry** (`R/registry.R`): every layer specification carries
   `spec$basis`; `layer_*()` dispatch to the implementation registered under
   that name. Built-in bases are found by name prefix in the namespace;
   other packages call `register_kan_basis(name, impl)`.
3. **Bases** (`R/basis_<name>.R`): one file per basis, all internal names
   prefixed `.<name>_`, so a basis can be added or removed without touching
   any other file.

## The basis interface (`impl`)

| field | signature | role |
|---|---|---|
| `layer_specs` | `(widths, ...)` -> list of one spec per layer | its formals are the arguments `thangld_kan()` accepts for the basis |
| `describe` | `(spec)` -> string | printing |
| `param_dims` | `(p, q, spec)` -> named list of dims | structure checks, load |
| `init` | `(p, q, spec)` -> named list of arrays | initialization (seed set by `thangld_kan()`) |
| `check_params` | `(par, spec)` | domain checks during training |
| `forward_r` | `(h, par, spec)` -> `list(out, cache)` | base R oracle and backend |
| `backward_r` | `(cache, g, par, spec)` -> `list(grads, dh)` | analytic gradients |
| `forward_torch` | `(h, par, spec)` -> tensor | torch backend (autograd) |
| `edge_r` | `(x, par, spec, i, j)` -> data frame with `phi` | edge evaluation and plots |
| `naive_forward` (optional) | `(h, par, spec)` | independent loop check |
| `basis_checks` (optional) | `()` -> list of `list(check, error, tol)` | basis-level checks |
| `validation_fixture` (optional) | `()` -> `list(model, x, y)` | fixture for the checks |

Missing optional fields make the corresponding checks *skipped*
(`pass = NA`), never passed. Thresholds:
basis 1e-10, forward 1e-8, gradients 1e-5 (fourth-order central
differences, h = 1e-4, 1e-5, 1e-6), save/load 1e-8, torch 1e-8.

## Plugging in splineKANr as `basis = "bspline"`

splineKANr 0.5.x has its own engine (`kan_fit()`, `kan_state`), so it stays a
separate package and nothing in thangldKANr depends on it. The plug-in is a
small adapter, planned for splineKANr 0.6.0 after 0.5.x is on CRAN and
thangldKANr is on CRAN:

1. splineKANr adds `Suggests: thangldKANr` and an exported (or internal)
   `kan_bspline_impl()` returning the list above, built from functions it
   already has:
   - `layer_specs(widths, grid_size = 5, spline_order = 3,
     grid_range = c(-1, 1), extrapolate = "linear", base_fun = "silu",
     bias = TRUE)` (the arguments of `kan_spec()`) stores the knot
     grid (not trained by the generic engine) in each spec;
   - parameters `coef[p, q, G + k]`, `scale_base[p, q]`, `scale_sp[p, q]`,
     `bias[q]` (the `kan_state` layout; `mask` is a constant of the spec),
     `init` = `kan_init_state()` for one layer;
   - `forward_r`/`backward_r` from `layer_forward_r()` and the analytic
     gradient code behind `kan_gradient_r()`; `forward_torch` from the
     tensor code behind `kan_module_from_state()`;
   - `basis_checks` = `kan_validate_basis()` and
     `kan_validate_derivative()` rows; `naive_forward` from the
     Cox-de Boor reference `kan_bspline_basis_r()`.
2. splineKANr registers the basis when both packages are loaded, from its
   `.onLoad()`:

   ```r
   .onLoad <- function(libname, pkgname) {
     reg <- function(...) thangldKANr::register_kan_basis(
       "bspline", kan_bspline_impl(), overwrite = TRUE)
     if (isNamespaceLoaded("thangldKANr")) reg()
     setHook(packageEvent("thangldKANr", "onLoad"), reg)
   }
   ```

   After `library(splineKANr)`, `thangld_kan(widths, basis = "bspline")`,
   mixed networks such as `basis = c("bspline", "chebyshev")`, both
   training backends, plots, save/load and
   `thangld_kan_validate(basis = "bspline")` work without any change to
   thangldKANr.
3. Features specific to splines that need data or change the parameter
   shapes (grid update `kan_update_grid()`, refinement, pruning,
   regularization) stay in splineKANr's own `kan_fit()`; the generic
   engine trains with a fixed grid. Whether to expose grid update in the
   generic engine later is a separate decision (it would need an optional
   `update_grid(par, spec, h)` field in the interface).
4. Alternative, once both are on CRAN: thangldKANr lists splineKANr in
   `Suggests` and registers `"bspline"` itself when splineKANr is
   installed. The interface is the same, so moving the adapter between the
   two packages does not change user code.

Names do not clash: splineKANr exports `kan_*` functions (`kan_fit`,
`kan_layer`, ...); thangldKANr exports `thangld_kan*`, `kan_bases`,
`kan_layer_bases`, `register_kan_basis`, `unregister_kan_basis` and the
basis functions (`chebyshev_basis`, `rbf_basis`, ...).

## Graph and temporal models

`thangld_graph_kan()` and `thangld_temporal_kan()` reuse the registry: their
layer specifications are built by the same code as `thangld_kan()`
(`build_layer_specs()`), so every basis, including a registered one, works
in them.

* Graph: layer `l` computes `KAN_l(A_hat H)` or, with `self_branch = TRUE`,
  `KAN_l([H, A_hat H])`; the graph engine (`R/graph_engine.R`) adds the
  aggregation and its adjoint around the generic `layer_*()` calls, and
  the basis sees `2 p` inputs with the self branch. Full-batch training
  with a loss on the training nodes only.
* Temporal: lag windows of a series (`thangld_temporal_kan_embed()`) are a
  numeric matrix, so the tabular engine is used unchanged; the model adds
  the windowing, optional log and differencing, and rolling-origin
  evaluation.
