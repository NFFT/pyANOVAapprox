Analysis
=======================

.. py:function:: get_variances(a, lam = None, Dict = False)

   This function returns the variances of the approximated ANOVA terms for all lam, if lam == None. Otherwise for the provided lam. Depending on Dict, it returns the approximated ANOVA terms as a vector or as a dict.

.. py:function:: get_GSI(a, lam = None, Dict = False)

   This function returns the global sensitivity indices of the approximation for all lam, if lam == None. Otherwise for the provided lam. Depending on Dict, it returns the approximated ANOVA terms as a vector or as a dict.

.. py:function:: get_AttributeRanking(a, lam = None)

   This function returns the attribute ranking of the approximation for all reg. parameters lam, if lam == None, as a dictionary of vectors of length ``a.d``. Otherwise for the provided lam as a vector of length ``a.d``.

.. py:function:: get_ShapleyValues(a, lam = None)

   This function returns the Shapley values of the approximation for all reg. parameters lam, if lam == None, as a dictionary of vectors of length ``a.d``. Otherwise for the provided lam as a vector of length ``a.d``.
