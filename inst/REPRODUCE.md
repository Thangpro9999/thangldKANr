# Reproducing the thangldKANr <-> Python comparisons

Python 3.12.16, PyTorch 2.14.1 (CPU), NumPy 2.5.4 (conda-forge), one thread,
float64. For each basis, clone the reference at the pinned commit
(`git clone <url>` then `git -C <dir> checkout <commit>`):

| basis | repository | commit | script |
|---|---|---|---|
| chebyshev | https://github.com/SynodicMonth/ChebyKAN | 5f7efdd18e749bcc99481bd87dc90bdeafb920d8 | chebykan_reference.py |
| rbf | https://github.com/ZiyaoLi/fast-kan | 17b65401c252334fffb5e63c9852dd8316d29e69 | fastkan_reference.py |
| legendre | https://github.com/Boris-73-TA/OrthogPolyKANs | 3b2760a7def56122dafaef53ad7ac3aa87804826 | legendrekan_reference.py |
| wavelet | https://github.com/zavareh1/Wav-KAN | f7ab5215e69b1a32e9e289159d2fd55d078fa051 | wavkan_reference.py |
| rational | https://github.com/ml-research/rational_activations | 2496d918d924b117ef06ec35bdca7f207720c62c | rationalkan_reference.py |
| hybrid | ChebyKAN and fast-kan (both clones, in that order) | as above | hybridkan_reference.py |
| jacobi | https://github.com/SpaceLearner/JacobiKAN | 51d160cfbf70914e4862e192ed3eef403dab893b | jacobikan_reference.py |
| relu | https://github.com/quiqi/relu_kan | 79eab7a27392eb5f33e119710f0e75ec902764c0 | relukan_reference.py |
| graph | ChebyKAN (as above) | 5f7efdd18e749bcc99481bd87dc90bdeafb920d8 | graphkan_reference.py |
| temporal | ChebyKAN (as above) | 5f7efdd18e749bcc99481bd87dc90bdeafb920d8 | chebykan_lagwindow_reference.py |

From `system.file("validation", package = "thangldKANr")`, for `<b>` (a basis name, `graph` or `temporal`):

    Rscript <b>/make_fixture.R work/<b>/fixture
    python -I python/<script> <clone(s)> work/<b>/fixture work/<b>/python_out
    Rscript <b>/compare_reference.R work/<b>/fixture work/<b>/python_out work/<b>/report
