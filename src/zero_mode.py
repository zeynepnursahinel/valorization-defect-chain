import numpy as np

from ssh_core import spectral_correlation_matrix


def trial_near_zero_profile(
    N: int,
    lam: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Analytical trial profile for the near-zero mode
    of an odd periodic SSH chain.
    """
    if N % 2 == 0:
        raise ValueError(
            "The trial near-zero profile is defined for odd N."
        )

    M = (N - 1) // 2

    t1 = 1.0 - lam
    t2 = 1.0 + lam

    phi = np.zeros(N, dtype=float)

    if lam > 0:
        r = t1 / t2

        for n in range(M + 1):
            phi[2 * n] = (-r) ** n

        for n in range(M):
            phi[2 * n + 1] = (
                (-1) ** (n + 1)
                * r ** (M - n)
            )

    elif lam < 0:
        s = t2 / t1

        for n in range(M + 1):
            phi[2 * n] = (
                (-1) ** n
                * s ** (M - n)
            )

        for n in range(M):
            phi[2 * n + 1] = (
                (-1) ** (n + 1)
                * s ** n
            )

    else:
        phi[:] = 1.0

    if normalize:
        norm = np.linalg.norm(phi)

        if norm == 0:
            raise ValueError(
                "The trial wavefunction has zero norm."
            )

        phi = phi / norm

    return phi


def numerical_near_zero_mode(
    N: int,
    lam: float,
    beta: float = 30.0,
    periodic: bool = True,
) -> dict:
    """
    Return the numerical eigenstate closest to zero energy.
    """
    data = spectral_correlation_matrix(
        N=N,
        lam=lam,
        beta=beta,
        periodic=periodic,
    )

    eigvals = data["eigvals"]
    eigvecs = data["eigvecs"]
    occupations = data["occupations"]

    k0 = int(np.argmin(np.abs(eigvals)))

    return {
        "mode_index": k0,
        "energy": float(eigvals[k0]),
        "occupation": float(occupations[k0]),
        "psi": eigvecs[:, k0],
    }


def compare_trial_and_numerical_mode(
    N: int,
    lam: float,
    beta: float = 30.0,
    periodic: bool = True,
) -> dict:
    """
    Compare numerical and analytical near-zero modes.
    """
    numerical = numerical_near_zero_mode(
        N=N,
        lam=lam,
        beta=beta,
        periodic=periodic,
    )

    psi = numerical["psi"]
    phi = trial_near_zero_profile(
        N=N,
        lam=lam,
        normalize=True,
    )

    if np.vdot(psi, phi).real < 0:
        phi = -phi

    wave_overlap = abs(np.vdot(psi, phi)) ** 2

    prob_psi = np.abs(psi) ** 2
    prob_phi = np.abs(phi) ** 2

    probability_overlap = (
        np.sum(
            np.sqrt(prob_psi * prob_phi)
        )
        ** 2
    )

    return {
        **numerical,
        "phi": phi,
        "wave_overlap": float(wave_overlap),
        "probability_overlap": float(
            probability_overlap
        ),
    }