# Example for using libnfftjulia directly from Python via ctypes.
#
# This mirrors the Julia example in:
#   nfft/simple_test_2d.jl
# and uses pyNFFT3 as reference for function signatures and memory handling.

import ctypes
import os
import platform
from pathlib import Path

import numpy as np


class nfft_plan(ctypes.Structure):
    pass


def detect_glibc_subdir() -> str:
    """Return the expected glibc subdir used by the Julia build layout."""
    if platform.system() != "Linux":
        return ""

    libc_version = platform.libc_ver()[1]
    try:
        major, minor = (int(x) for x in libc_version.split(".")[:2])
    except Exception:
        return "glibc2.40"

    if (major, minor) <= (2, 35):
        return "glibc2.22"
    return "glibc2.40"


def find_libnfftjulia() -> Path:
    """Locate libnfftjulia shared library.

    Search order:
    1) LIBNFFTJULIA_PATH environment variable
    2) Julia NFFT repo path used in this workspace
    3) pyNFFT3 packaged library path
    """
    env = os.getenv("LIBNFFTJULIA_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    system = platform.system()
    if system == "Windows":
        ending = ".dll"
    elif system == "Darwin":
        ending = ".dylib"
    else:
        ending = ".so"

    glibc_subdir = detect_glibc_subdir()

    candidates = [
        Path("/home/pasca/julia/nfftlibs/nfft-3.5.3-julia-openmp/nfft")
        / glibc_subdir
        / f"libnfftjulia{ending}",
        Path("/home/pasca/git/pyNFFT3/src/pyNFFT3/lib/AVX2")
        / glibc_subdir
        / f"libnfftjulia{ending}",
        Path("/home/pasca/git/pyNFFT3/src/pyNFFT3/lib/AVX")
        / glibc_subdir
        / f"libnfftjulia{ending}",
        Path("/home/pasca/git/pyNFFT3/src/pyNFFT3/lib/SSE2")
        / glibc_subdir
        / f"libnfftjulia{ending}",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find libnfftjulia. Set LIBNFFTJULIA_PATH to the .so/.dylib/.dll file."
    )


def build_api(lib: ctypes.CDLL):
    """Configure ctypes signatures (same API style as in pyNFFT3)."""
    lib.jnfft_alloc.restype = ctypes.POINTER(nfft_plan)

    lib.jnfft_init.argtypes = [
        ctypes.POINTER(nfft_plan),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]

    lib.jnfft_set_x.argtypes = [
        ctypes.POINTER(nfft_plan),
        np.ctypeslib.ndpointer(np.float64, flags="C"),
    ]
    lib.jnfft_set_x.restype = ctypes.POINTER(ctypes.c_double)

    lib.jnfft_set_fhat.argtypes = [
        ctypes.POINTER(nfft_plan),
        np.ctypeslib.ndpointer(np.complex128, ndim=1, flags="C"),
    ]
    lib.jnfft_set_fhat.restype = ctypes.POINTER(ctypes.c_double)

    lib.jnfft_set_f.argtypes = [
        ctypes.POINTER(nfft_plan),
        np.ctypeslib.ndpointer(np.complex128, ndim=1, flags="C"),
    ]
    lib.jnfft_set_f.restype = ctypes.POINTER(ctypes.c_double)

    lib.jnfft_trafo.argtypes = [ctypes.POINTER(nfft_plan)]
    lib.jnfft_trafo.restype = ctypes.POINTER(ctypes.c_double)

    lib.jnfft_adjoint.argtypes = [ctypes.POINTER(nfft_plan)]
    lib.jnfft_adjoint.restype = ctypes.POINTER(ctypes.c_double)

    lib.jnfft_finalize.argtypes = [ctypes.POINTER(nfft_plan)]

    if hasattr(lib, "nfft_get_num_threads"):
        lib.nfft_get_num_threads.restype = ctypes.c_int


def main():
    rng = np.random.default_rng(1234)

    # Same setup as Julia simple_test_2d.jl
    N = np.array([16, 8], dtype=np.int32)
    M = 10000
    D = len(N)
    Ns = int(np.prod(N))

    lib_path = find_libnfftjulia()
    lib = ctypes.CDLL(str(lib_path))
    build_api(lib)

    print("2d NFFT Test (Python ctypes + libnfftjulia)")
    print(f"Using library: {lib_path}")

    plan = lib.jnfft_alloc()

    n = (2 ** (np.ceil(np.log(N) / np.log(2)) + 1)).astype(np.int32)
    m = np.int32(8)

    # Values from Julia defaults in NFFT.jl
    f1_default = np.uint32(
        (1 << 0)  # PRE_PHI_HUT
        | (1 << 4)  # PRE_PSI
        | (1 << 6)  # MALLOC_X
        | (1 << 7)  # MALLOC_F_HAT
        | (1 << 8)  # MALLOC_F
        | (1 << 10)  # FFTW_INIT
        | (1 << 9)  # FFT_OUT_OF_PLACE
        | (1 << 11)  # NFFT_SORT_NODES
        | (1 << 12)  # NFFT_OMP_BLOCKWISE_ADJOINT
    )
    f2_default = np.uint32(1 << 6)  # FFTW_ESTIMATE

    lib.jnfft_init(
        plan,
        ctypes.c_int32(D),
        N.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(M),
        n.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(m),
        ctypes.c_uint32(int(f1_default)),
        ctypes.c_uint32(int(f2_default)),
    )

    if hasattr(lib, "nfft_get_num_threads"):
        print("Number of Threads:", lib.nfft_get_num_threads())

    # Julia uses A = rand(2,M) .- 0.5; here we store nodes as MxD (C order)
    X = np.ascontiguousarray(rng.random((M, D)) - 0.5, dtype=np.float64)
    lib.jnfft_set_x(plan, X)

    fhat = np.ascontiguousarray(
        rng.random(Ns) + 1.0j * rng.random(Ns), dtype=np.complex128
    )
    lib.jnfft_set_fhat(plan, fhat)

    # Forward transform
    f_ptr = lib.jnfft_trafo(plan)
    f_fast = np.ctypeslib.as_array(f_ptr, shape=(M * 2,)).view(np.complex128).copy()

    # Direct matrix (same frequency ordering as pyNFFT3/tests/NFFT_test.py)
    I = [
        [k0, k1]
        for k0 in range(-N[0] // 2, N[0] // 2)
        for k1 in range(-N[1] // 2, N[1] // 2)
    ]
    I = np.asarray(I, dtype=np.int32)

    F = np.exp(-2j * np.pi * (X @ I.T))
    f_direct = F @ fhat

    err_vec = f_direct - f_fast
    e2 = np.linalg.norm(err_vec) / np.linalg.norm(f_direct)
    einf = np.linalg.norm(err_vec, np.inf) / np.linalg.norm(fhat, 1)

    print("Forward E2:", e2)
    print("Forward Einf:", einf)

    # Adjoint transform
    lib.jnfft_set_f(plan, np.ascontiguousarray(f_fast, dtype=np.complex128))
    fhat_ptr = lib.jnfft_adjoint(plan)
    fhat_fast = np.ctypeslib.as_array(fhat_ptr, shape=(Ns * 2,)).view(np.complex128).copy()

    fhat_direct = F.conj().T @ f_fast
    err_vec_adj = fhat_direct - fhat_fast
    e2_adj = np.linalg.norm(err_vec_adj) / np.linalg.norm(fhat_direct)
    einf_adj = np.linalg.norm(err_vec_adj, np.inf) / np.linalg.norm(fhat, 1)

    print("Adjoint E2:", e2_adj)
    print("Adjoint Einf:", einf_adj)

    if e2 >= 1e-8 or einf >= 1e-8:
        raise RuntimeError("Forward errors are too large.")
    if e2_adj >= 1e-8 or einf_adj >= 1e-8:
        raise RuntimeError("Adjoint errors are too large.")

    lib.jnfft_finalize(plan)


if __name__ == "__main__":
    main()
