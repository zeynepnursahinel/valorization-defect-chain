from __future__ import annotations

import numpy as np

from ssh_core import (
    build_ssh_hamiltonian,
    spectral_correlation_matrix,
    fermi_occupation,
)


def build_reference_system(
    N: int,
    lam: float,
) -> dict:
    """
    Construct the reference and full Hamiltonians.

    H0:
        Open SSH chain.

    H:
        Odd periodic SSH chain.

    V:
        Perturbation that closes the open chain,

            H = H0 + V.
    """
    if N % 2 == 0:
        raise ValueError(
            "This reconstruction is intended for odd N."
        )

    H0 = build_ssh_hamiltonian(
        N=N,
        lam=lam,
        periodic=False,
    )

    H = build_ssh_hamiltonian(
        N=N,
        lam=lam,
        periodic=True,
    )

    V = H - H0

    return {
        "H0": H0,
        "H": H,
        "V": V,
    }


def fermi_complex(
    z: complex,
    beta: float,
) -> complex:
    """
    Fermi-Dirac function for a complex argument.
    """
    return 1.0 / (
        np.exp(beta * z) + 1.0
    )


def resolvent(
    H: np.ndarray,
    z: complex,
) -> np.ndarray:
    """
    Compute the resolvent

        G(z) = (z I - H)^(-1).

    The linear system is solved instead of explicitly
    constructing a matrix inverse.
    """
    H = np.asarray(H, dtype=complex)

    identity = np.eye(
        H.shape[0],
        dtype=complex,
    )

    return np.linalg.solve(
        z * identity - H,
        identity,
    )


def t_matrix(
    G0: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """
    Compute the T-matrix

        T(z) = (I - V G0(z))^(-1) V.
    """
    G0 = np.asarray(G0, dtype=complex)
    V = np.asarray(V, dtype=complex)

    identity = np.eye(
        G0.shape[0],
        dtype=complex,
    )

    return np.linalg.solve(
        identity - V @ G0,
        V,
    )


def ellipse_parameters(
    H0: np.ndarray,
    H: np.ndarray,
    beta: float,
    margin: float = 0.2,
    imag_height: float | None = None,
) -> tuple[float, float]:
    """
    Determine the semi-axes of the elliptical contour.

    The contour encloses the spectra of both H0 and H.

    The imaginary semi-axis is kept below the first
    Fermi-function pole at

        Im(z) = pi / beta.
    """
    eigvals_0 = np.linalg.eigvalsh(H0)
    eigvals = np.linalg.eigvalsh(H)

    spectral_radius = max(
        np.max(np.abs(eigvals_0)),
        np.max(np.abs(eigvals)),
    )

    horizontal_axis = (
        float(spectral_radius) + margin
    )

    if imag_height is None:
        vertical_axis = 0.5 * np.pi / beta
    else:
        vertical_axis = float(imag_height)

    if vertical_axis <= 0:
        raise ValueError(
            "imag_height must be positive."
        )

    if vertical_axis >= np.pi / beta:
        raise ValueError(
            "The contour crosses a Fermi-function pole. "
            "Choose imag_height < pi / beta."
        )

    return horizontal_axis, vertical_axis


def contour_reconstruction(
    N: int,
    lam: float,
    beta: float,
    n_theta: int = 2048,
    margin: float = 0.2,
    imag_height: float | None = None,
) -> dict:
    """
    Reconstruct the full correlation matrix using

        C_GF = C0_GF + delta_C_GF,

    where

        C0_GF
        =
        (1 / 2 pi i)
        integral f(z) G0(z) dz,

    and

        delta_C_GF
        =
        (1 / 2 pi i)
        integral f(z) G0(z) T(z) G0(z) dz.

    The reconstructed matrix is compared with the
    direct spectral result for the full Hamiltonian.
    """
    if beta <= 0:
        raise ValueError(
            "beta must be positive."
        )

    if n_theta < 4:
        raise ValueError(
            "n_theta must be at least 4."
        )

    system = build_reference_system(
        N=N,
        lam=lam,
    )

    H0 = system["H0"]
    H = system["H"]
    V = system["V"]

    horizontal_axis, vertical_axis = (
        ellipse_parameters(
            H0=H0,
            H=H,
            beta=beta,
            margin=margin,
            imag_height=imag_height,
        )
    )

    theta_values = np.linspace(
        0.0,
        2.0 * np.pi,
        n_theta,
        endpoint=False,
    )

    delta_theta = (
        2.0 * np.pi / n_theta
    )

    C0_GF = np.zeros(
        (N, N),
        dtype=complex,
    )

    delta_C_GF = np.zeros(
        (N, N),
        dtype=complex,
    )

    for theta in theta_values:
        z = (
            horizontal_axis * np.cos(theta)
            + 1j
            * vertical_axis
            * np.sin(theta)
        )

        dz_dtheta = (
            -horizontal_axis * np.sin(theta)
            + 1j
            * vertical_axis
            * np.cos(theta)
        )

        G0 = resolvent(
            H=H0,
            z=z,
        )

        T = t_matrix(
            G0=G0,
            V=V,
        )

        weight = (
            fermi_complex(
                z=z,
                beta=beta,
            )
            * dz_dtheta
            * delta_theta
            / (2.0j * np.pi)
        )

        C0_GF += weight * G0

        delta_C_GF += (
            weight
            * G0
            @ T
            @ G0
        )

    C_GF = C0_GF + delta_C_GF

    spectral_full = spectral_correlation_matrix(
        N=N,
        lam=lam,
        beta=beta,
        periodic=True,
    )

    spectral_reference = spectral_correlation_matrix(
        N=N,
        lam=lam,
        beta=beta,
        periodic=False,
    )

    C_diag = spectral_full["C"]
    C0_diag = spectral_reference["C"]

    difference = C_GF - C_diag
    reference_difference = C0_GF - C0_diag

    max_abs_error = float(
        np.max(
            np.abs(difference)
        )
    )

    frobenius_error = float(
        np.linalg.norm(
            difference,
            ord="fro",
        )
    )

    relative_frobenius_error = float(
        frobenius_error
        / np.linalg.norm(
            C_diag,
            ord="fro",
        )
    )

    reference_max_abs_error = float(
        np.max(
            np.abs(reference_difference)
        )
    )

    max_imaginary_part = float(
        np.max(
            np.abs(
                np.imag(C_GF)
            )
        )
    )

    return {
        "H0": H0,
        "H": H,
        "V": V,
        "C0_GF": C0_GF,
        "delta_C_GF": delta_C_GF,
        "C_GF": C_GF,
        "C0_diag": C0_diag,
        "C_diag": C_diag,
        "difference": difference,
        "max_abs_error": max_abs_error,
        "frobenius_error": frobenius_error,
        "relative_frobenius_error": (
            relative_frobenius_error
        ),
        "reference_max_abs_error": (
            reference_max_abs_error
        ),
        "max_imaginary_part": (
            max_imaginary_part
        ),
        "horizontal_axis": horizontal_axis,
        "vertical_axis": vertical_axis,
    }

def compute_deltaC_decomposition(
    N,
    lam,
    beta,
    n_theta=2048,
    margin=0.2,
    imag_height=None,
):
    """
    Decompose the defect-induced correlation correction into

        deltaC
        =
        deltaC_zero
        +
        deltaC_regular.

    The total correction is obtained from the Green-function
    contour reconstruction, while the near-zero contribution
    is extracted from the eigenstate closest to zero energy.
    """

    reconstruction = contour_reconstruction(
        N=N,
        lam=lam,
        beta=beta,
        n_theta=n_theta,
        margin=margin,
        imag_height=imag_height,
    )

    delta_total = reconstruction["delta_C_GF"]

    H = reconstruction["H"]
    H0 = reconstruction["H0"]

    ####################################################
    # Full chain
    ####################################################

    eigvals, eigvecs = np.linalg.eigh(H)

    idx = np.argmin(np.abs(eigvals))

    E = eigvals[idx]

    psi = eigvecs[:, idx]

    occ = fermi_occupation(
        np.array([E]),
        beta,
    )[0]

    C_zero_full = (
        occ
        * np.outer(
            psi,
            psi.conj(),
        )
    )

    ####################################################
    # Reference chain
    ####################################################

    eigvals0, eigvecs0 = np.linalg.eigh(H0)

    idx0 = np.argmin(np.abs(eigvals0))

    E0 = eigvals0[idx0]

    psi0 = eigvecs0[:, idx0]

    occ0 = fermi_occupation(
        np.array([E0]),
        beta,
    )[0]

    C_zero_ref = (
        occ0
        * np.outer(
            psi0,
            psi0.conj(),
        )
    )

    ####################################################
    # decomposition
    ####################################################

    delta_zero = (
        C_zero_full
        -
        C_zero_ref
    )

    delta_regular = (
        delta_total
        -
        delta_zero
    )

    return {

        "delta_total": delta_total,

        "delta_zero": delta_zero,

        "delta_regular": delta_regular,

        "H": H,
        "H0": H0,

        "E_zero": E,
        "E_zero_ref": E0,

        "psi_zero": psi,
        "psi_zero_ref": psi0,

    }
