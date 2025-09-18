Approximation
=======================

.. py:class:: approx(X, y, U, N, basis="cos")

   A class to hold the scattered data function approximation.

   This class represents an approximation object constructed from scattered data nodes, function values, an ANOVA decomposition class, and a choice of basis functions. It supports multiple bases and regularization parameters.

   .. rubric:: Attributes:

   .. py:attribute:: basis

      **Type:** `str`

      Basis of the function space. Supported values are:

      - `"per"`: exponential functions
      - `"cos"`: cosine functions
      - `"cheb"`: Chebyshev basis
      - `"std"`: transformed exponential functions
      - `"chui1"`: Haar wavelets
      - `"chui2"`: Chui-Wang wavelets of order 2
      - `"chui3"`: Chui-Wang wavelets of order 3
      - `"chui4"`: Chui-Wang wavelets of order 4

   .. py:attribute:: X

      **Type:** `Numpy array wit dtype float`

      The scattered data nodes with `M` rows and `d` columns.

   .. py:attribute:: y

      **Type:** `Numpy array wit dtype float or complex`

      A vector of `M` function values. Complex-valued for `basis = "per"`, real-valued otherwise.

   .. py:attribute:: U

      **Type:** `List[Tuple[int]]`

      A vector containing subsets of coordinate indices representing the ANOVA decomposition.

   .. py:attribute:: N

      **Type:** `List[int]`

      Bandwidths for each ANOVA term.

   .. py:attribute:: trafo

      **Type:** `GroupedTransform`

      Holds the grouped transformation.

   .. py:attribute:: fc

      **Type:** `Dict[float, GroupedCoefficients]`

      holds the GroupedCoefficients after approximation for every different regularization parameters

   .. rubric:: Constructor:

   .. py:method:: approx(X, y, U, N, basis="cos")


   .. rubric:: Additional Constructor:

   .. py:method:: approx(X, y, ds, N, basis="cos")


   .. rubric:: Functions:

   .. py:method:: approximate(self, lam, max_iter=50, weights=None, verbose=False, solver=None, tol=1e-8)

      If ``lam`` is a ``np.ndarray`` of dtype float, this function computes the approximation for the regularization parameters contained in ``lam``.

      If ``lam`` is a float, this function computes the approximation for the regularization parameter ``lam``.

   .. py:method:: evaluate(self, lam=None, X=None)

      This function evaluates the approximation with optional node matrix ``X`` and regularization ``lam``.

      - If both ``X`` and ``lam`` are given: evaluate at ``X`` for specific ``lam``.
      - If only ``X`` is given: evaluate at ``X`` for all ``lam``.
      - If only ``lam`` is given: evaluate at ``self.X`` for specific ``lam``.
      - If neither are given: evaluate at ``self.X`` for all ``lam``.

   .. py:method:: evaluateANOVAterms(self, X, lam=None)

      This function evaluates the single ANOVA terms of the approximation on the nodes of matrix ``X`` and regularization ``lam``.

      - If ``lam`` is given: evaluate at ``X`` for specific ``lam``.
      - If ``lam`` is not given: evaluate at ``X`` for all ``lam``.

   .. py:method:: evaluateSHAPterms(self, X, lam=None)

      This function evaluates for each dimension the Shapley contribution to the overall approximation on the nodes of matrix ``X`` and regularization ``lam``.

      - If ``lam`` is given: evaluate at ``X`` for specific ``lam``.
      - If ``lam`` is not given: evaluate at ``X`` for all ``lam``.




      
