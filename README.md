# pyANOVAapprox

[![](https://github.com/NFFT/pyANOVAapprox/actions/workflows/ci.yml/badge.svg)](https://github.com/NFFT/pyANOVAapprox/actions/workflows/ci.yml)

This package provides a framework for the method ANOVAapprox to approximate high-dimensional functions with a low superposition dimension or a sparse ANOVA decomposition from scattered data. The method has been dicussed and applied in the following articles/preprints:

<ul>
  <li>D. Potts and M. Schmischke <br> 
  <b>Interpretable transformed ANOVA approximation on the example of the prevention of forest fires</b> <br>
  <a href="https://arxiv.org/abs/2110.07353">arXiv</a>, <a href="https://www-user.tu-chemnitz.de/~mischmi/papers/transformedanova.pdf">PDF</a></li>
  <li>F. Bartel, D. Potts und M. Schmischke <br> 
  <b>Grouped transformations and Regularization in high-dimensional explainable ANOVA approximation</b> <br>
  SIAM Journal on Scientific Computing (accepted) <br>
  <a href="https://arxiv.org/abs/2010.10199">arXiv</a>, <a href="https://www-user.tu-chemnitz.de/~mischmi/papers/groupedtransforms.pdf">PDF</a></li>
  <li>D. Potts and M. Schmischke <br> 
  <b>Interpretable approximation of high-dimensional data</b> <br>
  SIAM Journal on Mathematics of Data Science (accepted) <br>
  <a href="https://arxiv.org/abs/2103.13787">arXiv</a>, <a href="https://www-user.tu-chemnitz.de/~mischmi/papers/attributeranking.pdf">PDF</a>, <a href="https://github.com/NFFT/AttributeRankingExamples">Software</a></li>
  <li>D. Potts and M. Schmischke <br> 
  <b>Learning multivariate functions with low-dimensional structures using polynomial bases</b><br>
  Journal of Computational and Applied Mathematics 403, 113821, 2021<br>
  <a href="https://doi.org/10.1016/j.cam.2021.113821">DOI</a>, <a href="https://arxiv.org/abs/1912.03195">arXiv</a>, <a href="https://www-user.tu-chemnitz.de/~mischmi/papers/anovacube.pdf">PDF</a></li>
  <li>D. Potts and M. Schmischke <br> 
  <b>Approximation of high-dimensional periodic functions with Fourier-based methods</b><br>
  SIAM Journal on Numerical Analysis 59 (5), 2393-2429, 2021<br>
  <a href="https://doi.org/10.1137/20M1354921">DOI</a>, <a href="https://arxiv.org/abs/1907.11412">arXiv</a>, <a href="https://www-user.tu-chemnitz.de/~mischmi/papers/anovafourier.pdf">PDF</a></li>
<li>L. Lippert, D. Potts and T. Ullrich <br> 
  <b>Fast Hyperbolic Wavelet Regression meets ANOVA</b><br>
  ArXiv: 2108.13197<br>
  <a href="https://arxiv.org/abs/2108.13197">arXiv</a>, <a href="https://www-user.tu-chemnitz.de/~lipl/paper/HWR.pdf">PDF</a></li>


</ul>

`pyANOVAapprox` provides the following functionality:
- approximation of high-dimensional periodic and nonperiodic functions with a sparse ANOVA decomposition
- analysis tools for interpretability (global sensitvitiy indices, attribute ranking, shapley values)

## Getting started

The [pyANOVAapprox package](https://pypi.org/project/pyANOVAapprox/) can be installed via pip:

```
pip install -i https://test.pypi.org/simple/ pyANOVAapprox
```

Read the [documentation](https://nfft.github.io/pyANOVAapprox/) for specific usage information.

Requirements
------------

- Python 3.8 or greater
- pyGroupedTransforms 0.1.0 or greater
- NumPy 2.0.0 or greater
- SciPy 1.16.0 or greater
