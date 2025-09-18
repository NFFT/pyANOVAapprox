import numpy as np


def B2(x: float) -> float:
    norm_b2_cheb = np.sqrt((-423 * np.sqrt(3) + 920 * np.pi) / (4096 * np.pi))

    if x < -1 or x > 1:
        raise ValueError("Out of bounds.")

    if x < -9 / 2:
        y = 0.0
    elif x < -5 / 2:
        y = 1 / 32 * (2 * x + 9) ** 2
    elif x < -1 / 2:
        y = 3 / 16 - 3 / 4 * x - 1 / 4 * x**2
    elif x < 3 / 2:
        y = 9 / 32 - 3 / 8 * x + 1 / 8 * x**2
    else:
        y = 0.0

    return y / norm_b2_cheb


def B4(x: float) -> float:
    norm_b4_cheb = np.sqrt(
        (-36614943 * np.sqrt(3) + 218675240 * np.pi) / (6341787648 * np.pi)
    )

    if x < -1 or x > 1:
        raise ValueError("Out of bounds.")

    if x < -15 / 2:
        y = 0.0
    elif x < -11 / 2:
        y = 1 / 6144 * (2 * x + 15) ** 4
    elif x < -7 / 2:
        y = -5645 / 1536 - 205 / 48 * x - 95 / 64 * x**2 - 5 / 24 * x**3 - 1 / 96 * x**4
    elif x < -3 / 2:
        y = 715 / 3072 + 25 / 128 * x + 55 / 128 * x**2 + 5 / 32 * x**3 + 1 / 64 * x**4
    elif x < 1 / 2:
        y = 155 / 1536 - 5 / 32 * x + 5 / 64 * x**2 - 1 / 96 * x**4
    elif x < 5 / 2:
        y = 1 / 6144 * (2 * x - 5) ** 4
    else:
        y = 0.0

    return y / norm_b4_cheb


def B2_chat(k: int) -> float:
    norm_b2_cheb = np.sqrt((-423 * np.sqrt(3) + 920 * np.pi) / (4096 * np.pi))

    if k < 0:
        raise ValueError("Out of bounds.")

    if k == 0:
        fc = 1 / 4 + (9 * np.sqrt(3)) / (64 * np.pi)
    elif k == 1:
        fc = -1 / 2 + (9 * np.sqrt(3)) / (32 * np.pi)
    elif k == 2:
        fc = (9 * np.sqrt(3)) / (128 * np.pi)
    else:
        fc = (
            9 * np.sqrt(3) * k * np.cos((2 * k * np.pi) / 3)
            - 9 * (-2 + k**2) * np.sin((2 * k * np.pi) / 3)
        ) / (8 * k * (4 - 5 * k**2 + k**4) * np.pi)

    if k != 0:
        fc /= np.sqrt(2)

    return fc / norm_b2_cheb


def B4_chat(k: int) -> float:
    norm_b4_cheb = np.sqrt(
        (-36614943 * np.sqrt(3) + 218675240 * np.pi) / (6341787648 * np.pi)
    )

    if k < 0:
        raise ValueError("Out of bounds.")

    if k == 0:
        fc = 2603 / 18432 - 75 / 8192 * np.sqrt(3) / np.pi
    elif k == 1:
        fc = -95 / 576 + 33 * np.sqrt(3) / (2048 * np.pi)
    elif k == 2:
        fc = 181 / 4608 - 39 * np.sqrt(3) / (4096 * np.pi)
    elif k == 3:
        fc = (5 * (-14 + (27 * np.sqrt(3)) / np.pi)) / 32256
    elif k == 4:
        fc = -7 / 9216 - 93 * np.sqrt(3) / (114688 * np.pi)
    else:
        fc = (
            (900 * np.sqrt(3) * k * (-9 + k**2) * np.cos(k * np.pi) / 3)
            + 90 * (152 - 75 * k**2 + 3 * k**4) * np.sin(k * np.pi / 3)
        ) / (
            768 * k * (-16 + k**2) * (-9 + k**2) * (-4 + k**2) * (-1 + k**2) * np.pi
        )

    if k != 0:
        fc /= np.sqrt(2)

    return fc / norm_b4_cheb


Terms = {1: (0, 4), 2: (1, 5), 3: (2, 6), 4: (3, 7)}


def f(x: np.ndarray) -> float:
    if x.shape[0] != 8:
        raise ValueError("Vector has to be 8-dimensional")

    y = 0.0
    for i in range(1, 5):
        y += B2(x[Terms[i][0]]) * B4(x[Terms[i][1]])
    return y


def fc(k: np.ndarray) -> float:
    if k.shape[0] != 8:
        raise ValueError("Index has to be 8-dimensional")

    ind = np.array([0 if ki == 0 else 1 for ki in k])
    Terms_Support = {
        i: int(np.sum(ind) == np.sum(ind[list(Terms[i])])) for i in range(1, 5)
    }

    y = 0.0
    for i in range(1, 5):
        y += Terms_Support[i] * B2_chat(k[(Terms[i])[0]]) * B4_chat(k[(Terms[i])[1]])
    return y


AS = [
    (),
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    (5,),
    (6,),
    (7,),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def norm() -> float:
    return np.sqrt(4 + 12 * B2_chat(0) ** 2 * B4_chat(0) ** 2)


def inverse_distribution(x: float) -> float:
    return np.sin(np.pi * (x - 0.5))


def generateData(M: int) -> tuple[np.ndarray, np.ndarray]:
    X = 2 * np.random.rand(8, M) - 1
    X = inverse_distribution(X)
    y = np.array([f(X[:, i]) for i in range(M)])
    return X, y
