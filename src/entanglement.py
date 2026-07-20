import numpy as np
from scipy.linalg import sqrtm


def rho_from_two_site_correlation(
    C_A: np.ndarray,
) -> np.ndarray:
    """
    Construct the 4x4 two-mode fermionic RDM from a
    2x2 number-conserving Gaussian correlation matrix.

    Basis
    -----
    |00>, |01>, |10>, |11>
    """
    C_A = np.asarray(C_A, dtype=complex)

    if C_A.shape != (2, 2):
        raise ValueError(
            "C_A must have shape (2, 2)."
        )

    C_A = 0.5 * (C_A + C_A.conj().T)

    a = np.real_if_close(C_A[0, 0])
    b = np.real_if_close(C_A[1, 1])
    g = C_A[0, 1]

    rho00 = (
        (1.0 - a) * (1.0 - b)
        - abs(g) ** 2
    )

    rho01 = (
        b * (1.0 - a)
        + abs(g) ** 2
    )

    rho10 = (
        a * (1.0 - b)
        + abs(g) ** 2
    )

    rho11 = (
        a * b
        - abs(g) ** 2
    )

    rho = np.array(
        [
            [rho00, 0, 0, 0],
            [0, rho01, g, 0],
            [0, np.conj(g), rho10, 0],
            [0, 0, 0, rho11],
        ],
        dtype=complex,
    )

    rho = 0.5 * (
        rho + rho.conj().T
    )

    rho[np.abs(rho) < 1e-14] = 0.0

    trace = np.trace(rho)

    if not np.isclose(trace, 1.0):
        rho = rho / trace

    return rho


def concurrence_wootters(
    rho: np.ndarray,
) -> float:
    """
    Compute Wootters concurrence.
    """
    rho = np.asarray(rho, dtype=complex)

    if rho.shape != (4, 4):
        raise ValueError(
            "rho must have shape (4, 4)."
        )

    sigma_y = np.array(
        [
            [0, -1j],
            [1j, 0],
        ],
        dtype=complex,
    )

    spin_flip = np.kron(
        sigma_y,
        sigma_y,
    )

    rho_tilde = (
        spin_flip
        @ rho.conj()
        @ spin_flip
    )

    sqrt_rho = sqrtm(rho)

    R = sqrtm(
        sqrt_rho
        @ rho_tilde
        @ sqrt_rho
    )

    eigenvalues = np.linalg.eigvals(R)
    eigenvalues = np.real_if_close(
        eigenvalues
    ).real

    eigenvalues = np.sort(
        np.maximum(eigenvalues, 0.0)
    )[::-1]

    concurrence = (
        eigenvalues[0]
        - eigenvalues[1]
        - eigenvalues[2]
        - eigenvalues[3]
    )

    return float(max(0.0, concurrence))


def concurrence_from_two_site_correlation(
    C_A: np.ndarray,
) -> float:
    """
    Convenience wrapper:
        C_A -> rho_A -> concurrence
    """
    rho = rho_from_two_site_correlation(C_A)

    return concurrence_wootters(rho)