import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.linalg import eigh, sqrtm
import string
from qutip import Qobj, ptrace
from scipy.special import expit
from pathlib import Path





def build_ssh_hamiltonian(
    N: int,
    lam: float,
    periodic: bool = False,
) -> np.ndarray:
    """
    Construct the SSH single-particle Hamiltonian.

    Parameters
    ----------
    N
        Number of lattice sites.
    lam
        Dimerization parameter.
    periodic
        If True, impose periodic boundary conditions.
    """
    if N < 2:
        raise ValueError("N must be at least 2.")

    H = np.zeros((N, N), dtype=float)

    t1 = 1.0 - lam
    t2 = 1.0 + lam

    max_i = N if periodic else N - 1

    for i in range(max_i):
        j = (i + 1) % N
        hopping = t1 if i % 2 == 0 else t2
        H[i, j] = hopping
        H[j, i] = hopping

    return H


def fermi_occupation(
    energies: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Numerically stable Fermi-Dirac occupations.
    """
    energies = np.asarray(energies, dtype=float)

    if beta <= 0:
        raise ValueError("beta must be positive.")

    return expit(-beta * energies)


def spectral_correlation_matrix(
    N: int,
    lam: float,
    beta: float,
    periodic: bool = True,
) -> dict:
    """
    Compute the correlation matrix from spectral decomposition.

    Returns
    -------
    dict containing
        H, eigvals, eigvecs, occupations, C
    """
    H = build_ssh_hamiltonian(
        N=N,
        lam=lam,
        periodic=periodic,
    )

    eigvals, eigvecs = eigh(H)

    occupations = fermi_occupation(
        energies=eigvals,
        beta=beta,
    )

    C = eigvecs @ np.diag(occupations) @ eigvecs.T

    return {
        "H": H,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "occupations": occupations,
        "C": C,
    }


def extract_two_site_block(
    C: np.ndarray,
    pair: tuple[int, int],
) -> np.ndarray:
    """
    Extract the 2x2 correlation block for a selected site pair.
    """
    i, j = pair

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix.")

    N = C.shape[0]

    if not (0 <= i < N and 0 <= j < N):
        raise IndexError("Pair indices are outside the matrix range.")

    return C[np.ix_([i, j], [i, j])]

