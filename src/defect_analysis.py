import numpy as np

from ssh_core import (
    spectral_correlation_matrix,
    extract_two_site_block,
)

from zero_mode import (
    trial_near_zero_profile,
)

from entanglement import (
    concurrence_from_two_site_correlation,
)


KS = np.linspace(
    0.0,
    2.0 * np.pi,
    8192,
    endpoint=False,
)


def ssh_couplings(
    lam: float,
) -> tuple[float, float]:
    return 1.0 - lam, 1.0 + lam


def bloch_offdiagonal(
    lam: float,
    k: np.ndarray,
) -> np.ndarray:
    t1, t2 = ssh_couplings(lam)

    return t1 + t2 * np.exp(-1j * k)


def eta_continuum(
    lam: float,
    beta: float,
    r: int,
    ks: np.ndarray = KS,
) -> float:
    h_k = bloch_offdiagonal(
        lam=lam,
        k=ks,
    )

    abs_h = np.abs(h_k)
    phase = np.angle(h_k)

    weight = np.tanh(
        0.5 * beta * abs_h
    )

    eta = np.mean(
        weight
        * np.cos(r * ks + phase)
    )

    return float(np.real_if_close(eta))


def bulk_two_site_correlation_matrix(
    lam: float,
    beta: float,
    pair: tuple[int, int],
) -> np.ndarray:
    """
    Translationally invariant bulk two-site
    correlation matrix for the corresponding
    SSH bond parity.
    """
    i, j = pair

    if (j - i) % 2 == 0:
        raise ValueError(
            "This function currently assumes "
            "nearest-neighbor pairs."
        )

    r = 0 if i % 2 == 0 else 1

    eta = eta_continuum(
        lam=lam,
        beta=beta,
        r=r,
    )

    g_bulk = -0.5 * eta

    return np.array(
        [
            [0.5, g_bulk],
            [g_bulk, 0.5],
        ],
        dtype=float,
    )

def two_site_defect_decomposition(
    N: int,
    lam: float,
    beta: float,
    pair: tuple[int, int],
    periodic: bool = True,
) -> dict:
    """
    Linear decomposition at correlation-matrix level:

        C_A^defect
        =
        C_A^bulk
        +
        C_A^zero
        +
        R_A
    """
    spectral = spectral_correlation_matrix(
        N=N,
        lam=lam,
        beta=beta,
        periodic=periodic,
    )

    C_defect = extract_two_site_block(
        C=spectral["C"],
        pair=pair,
    )

    C_bulk = bulk_two_site_correlation_matrix(
        lam=lam,
        beta=beta,
        pair=pair,
    )

    eigvals = spectral["eigvals"]
    occupations = spectral["occupations"]

    k0 = int(
        np.argmin(
            np.abs(eigvals)
        )
    )

    E0 = float(eigvals[k0])
    f0 = float(occupations[k0])

    phi = trial_near_zero_profile(
        N=N,
        lam=lam,
        normalize=True,
    )

    i, j = pair
    phi_A = phi[[i, j]]

    C_zero = (
        f0
        * np.outer(
            phi_A,
            phi_A.conj(),
        )
    )

    C_approx = C_bulk + C_zero

    residual = (
        C_defect
        - C_approx
    )

    reconstruction_error = np.max(
        np.abs(
            C_defect
            - (
                C_bulk
                + C_zero
                + residual
            )
        )
    )

    return {
        "C_defect": C_defect,
        "C_bulk": C_bulk,
        "C_zero": C_zero,
        "C_approx": C_approx,
        "residual": residual,
        "E0": E0,
        "f0": f0,
        "reconstruction_error": float(
            reconstruction_error
        ),
    }

def compare_pair_concurrences(
    N: int,
    lam: float,
    beta: float,
    pair: tuple[int, int],
    periodic: bool = True,
) -> dict:
    """
    Compare concurrence for:

    1. uniform bulk,
    2. bulk + near-zero-mode approximation,
    3. full defect chain.
    """
    data = two_site_defect_decomposition(
        N=N,
        lam=lam,
        beta=beta,
        pair=pair,
        periodic=periodic,
    )

    concurrence_bulk = (
        concurrence_from_two_site_correlation(
            data["C_bulk"]
        )
    )

    concurrence_approx = (
        concurrence_from_two_site_correlation(
            data["C_approx"]
        )
    )

    concurrence_defect = (
        concurrence_from_two_site_correlation(
            data["C_defect"]
        )
    )

    return {
        **data,
        "concurrence_bulk": concurrence_bulk,
        "concurrence_approx": concurrence_approx,
        "concurrence_defect": concurrence_defect,
        "concurrence_residual": (
            concurrence_defect
            - concurrence_approx
        ),
    }