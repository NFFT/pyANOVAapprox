#!/usr/bin/env python
# coding: utf-8

# In[ ]:

AS = [(), (0,), (1,), (2,), (3,), (4,), (5,), (0, 2), (1, 4), (3, 5)]

import numpy as np

# Coefficients C[0] = C₁, C[1] = C₂, C[2] = C₃
# (Julia is 1-based; Python is 0-based)
C = np.array([np.sqrt(0.75), np.sqrt(315 / 604), np.sqrt(277200 / 655177)])

# --- B-spline definitions ---


def b_spline_2(x):
    C_2 = C[0]
    if 0.0 <= x < 0.5:
        return C_2 * 4 * x
    elif 0.5 <= x < 1.0:
        return C_2 * 4 * (1 - x)
    else:
        raise ValueError("B-spline 2: x out of range [0,1)")


def b_spline_4(x):
    C_4 = C[1]
    if 0.0 <= x < 0.25:
        return C_4 * (128 / 3) * x**3
    elif 0.25 <= x < 0.5:
        return C_4 * (8 / 3 - 32 * x + 128 * x**2 - 128 * x**3)
    elif 0.5 <= x < 0.75:
        return C_4 * (-88 / 3 - 256 * x**2 + 160 * x + 128 * x**3)
    elif 0.75 <= x < 1.0:
        return C_4 * (128 / 3 - 128 * x + 128 * x**2 - (128 / 3) * x**3)
    else:
        raise ValueError("B-spline 4: x out of range [0,1)")


def b_spline_6(x):
    C_6 = C[2]
    if 0.0 <= x < 1.0 / 6:
        return C_6 * (1944 / 5) * x**5
    elif 1.0 / 6 <= x < 2.0 / 6:
        return C_6 * (
            3 / 10 - 9 * x + 108 * x**2 - 648 * x**3 + 1944 * x**4 - 1944 * x**5
        )
    elif 2.0 / 6 <= x < 0.5:
        return C_6 * (
            -237 / 10 + 351 * x - 2052 * x**2 + 5832 * x**3 - 7776 * x**4 + 3888 * x**5
        )
    elif 0.5 <= x < 4.0 / 6:
        return C_6 * (
            2193 / 10
            + 7668 * x**2
            - 2079 * x
            + 11664 * x**4
            - 13608 * x**3
            - 3888 * x**5
        )
    elif 4.0 / 6 <= x < 5.0 / 6:
        return C_6 * (
            -5487 / 10
            + 3681 * x
            - 9612 * x**2
            + 12312 * x**3
            - 7776 * x**4
            + 1944 * x**5
        )
    elif 5.0 / 6 <= x < 1.0:
        return C_6 * (
            1944 / 5
            - 1944 * x
            + 3888 * x**2
            - 3888 * x**3
            + 1944 * x**4
            - (1944 / 5) * x**5
        )
    else:
        raise ValueError("B-spline 6: x out of range [0,1)")


# --- Block structure ---

m1 = [0, 2]  # 1-based [1, 3]
m2 = [1, 4]  # 2-based [2, 5]
m3 = [3, 5]  # 3-based [4, 6]

# --- Transformed function f(x) ---


def trans(x):
    return x + 1 if x < 0 else x


def f(x):
    x = np.asarray(x)
    if x.shape != (6,):
        raise ValueError("f(x): Input must be 6-dimensional.")
    if np.any(x < -0.5) or np.any(x > 0.5):
        raise ValueError("f(x): All entries must be in [-0.5, 0.5].")

    xT = np.where(x < 0, x + 1, x)  # vectorized 'trans'
    return (
        np.prod([b_spline_2(xT[i]) for i in m1])
        + np.prod([b_spline_4(xT[i]) for i in m2])
        + np.prod([b_spline_6(xT[i]) for i in m3])
    )


# --- sinc and b(k, r) function ---


def sinc(x):
    return 1.0 if x == 0.0 else np.sin(x) / x


def b(k, r):
    idx = r // 2 - 1  # Adjust for Python 0-based indexing
    return C[idx] * (sinc(np.pi * k / r) ** r) * np.cos(np.pi * k)


# --- Fourier coefficients fc(k) ---


def fc(k):
    if len(k) != 6:
        raise ValueError("fc(k): k must be 6-dimensional.")

    ind = np.array([int(ki != 0) for ki in k])

    b2_block = np.sum(ind) == np.sum(ind[m1])
    b4_block = np.sum(ind) == np.sum(ind[m2])
    b6_block = np.sum(ind) == np.sum(ind[m3])

    val = 0.0
    if b2_block:
        val += np.prod([b(k[i], 2) for i in m1])
    if b4_block:
        val += np.prod([b(k[i], 4) for i in m2])
    if b6_block:
        val += np.prod([b(k[i], 6) for i in m3])
    return val


# --- Norm computation ---


def norm():
    result = 3.0
    for i in range(1, 3):  # i = 1, 2
        for j in range(i + 1, 4):  # j = 2, 3
            result += 2 * b(0, 2 * i) ** 2 * b(0, 2 * j) ** 2
    return np.sqrt(result)
