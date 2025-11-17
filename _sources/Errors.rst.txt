Errors
============

.. py:function:: get_l2_error(a, X=None, y=None, lam=None)

   Computes the relative ``\ell_2`` error for an approx object ``a``.

   - If only ``a`` and ``lam`` are provided,  
     this function computes the relative ``\ell_2`` error on the training nodes for the specific regularization parameter ``lam``.

   - If ``a``, ``X``, ``y``, and ``lam`` are provided,  
     this function computes the relative ``\ell_2`` error on the given data ``X`` and ``y`` for the specified regularization parameter ``lam``.

   - If only ``a`` is provided,  
     this function computes the relative ``\ell_2`` error on the training nodes for all available regularization parameters.

   - If ``a``, ``X``, and ``y`` are provided,  
     this function computes the relative ``\ell_2`` error on the given data ``X`` and ``y`` for all available regularization parameters.

   Returns either a float (for a single ``lam``) or a dictionary mapping ``lam`` values to errors.


.. py:function:: get_mse(a, X=None, y=None, lam=None)

   Computes the mean square error (MSE) for an approx object ``a``.

   - If only ``a`` and ``lam`` are provided,  
     this function computes the mean square error on the training nodes for a specific regularization parameter ``lam``.

   - If ``a``, ``X``, ``y``, and ``lam`` are provided,  
     this function computes the mean square error on the given data ``X`` and ``y`` for the specified regularization parameter ``lam``.

   - If only ``a`` is provided,  
     this function computes the mean square error on the training nodes for all available regularization parameters.

   - If ``a``, ``X``, and ``y`` are provided,  
     this function computes the mean square error on the given data ``X`` and ``y`` for all available regularization parameters.

   Returns either a float (for a single ``lam``) or a dictionary mapping each ``lam`` to its corresponding MSE value.


.. py:function:: get_mad(a, X=None, y=None, lam=None)

   Computes the mean absolute deviation (MAD) for an approx object ``a``.

   - If only ``a`` and ``lam`` are provided,  
     this function computes the mean absolute deviation on the training nodes for a specific regularization parameter ``lam``.

   - If ``a``, ``X``, ``y``, and ``lam`` are provided,  
     this function computes the mean absolute deviation on the given data ``X`` and ``y`` for the specified regularization parameter ``lam``.

   - If only ``a`` is provided,  
     this function computes the mean absolute deviation on the training nodes for all available regularization parameters.

   - If ``a``, ``X``, and ``y`` are provided,  
     this function computes the mean absolute deviation on the given data ``X`` and ``y`` for all available regularization parameters.

   Returns a single MAD value (float), if ``lam`` is provided, or a dictionary mapping each ``lam`` to its corresponding MAD.


.. py:function:: get_L2_error(a, norm, bc_fun, lam=None)

   Computes the relative L2 error of a function approximation for an ``approx`` object ``a``.

   - If ``a``, ``norm``, ``bc_fun``, and ``lam`` are provided,  
     this function computes the relative L2 error for a specific regularization parameter ``lam``.

   - If only ``a``, ``norm``, and ``bc_fun`` are provided,  
     this function computes the relative L2 error for all available regularization parameters.
