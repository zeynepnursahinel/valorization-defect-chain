from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zero_mode import compare_trial_and_numerical_mode
from defect_analysis import (
    two_site_defect_decomposition,
    compare_pair_concurrences,
)

# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

FIG_ROOT = PROJECT_ROOT / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

FIG_DIR = FIG_ROOT / "defect"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def _prepare_output_path(savepath):
    """
    Resolve an output path relative to figures/defect.

    Examples
    --------
    savepath=None
        Figure is not saved.

    savepath="correlation_decomposition.pdf"
        Saved under figures/defect/.

    savepath=Path("other_folder/figure.pdf")
        Absolute paths are preserved; relative paths are interpreted
        relative to figures/defect/.
    """
    if savepath is None:
        return None

    savepath = Path(savepath)

    if not savepath.is_absolute():
        savepath = FIG_DIR / savepath

    savepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return savepath.resolve()


def _make_axes(
    nplots,
    ncols=2,
    figsize_per_panel=(4.8, 3.5),
    sharex=True,
    sharey=False,
):
    """
    Create a subplot grid and return flattened axes.
    """
    if nplots < 1:
        raise ValueError("nplots must be at least 1.")

    ncols = min(ncols, nplots)
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize_per_panel[0] * ncols,
            figsize_per_panel[1] * nrows,
        ),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )

    axes = axes.ravel()

    for ax in axes[nplots:]:
        ax.axis("off")

    return fig, axes


def _finish_figure(
    fig,
    axes,
    nplots,
    title=None,
    savepath=None,
    legend=True,
    legend_outside=True,
):
    """
    Add common legend, title, layout and optional saving.
    """
    if title is not None:
        fig.suptitle(title, fontsize=14)

    if legend and nplots > 0:
        handles, labels = axes[0].get_legend_handles_labels()

        if handles:
            if legend_outside:
                fig.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    frameon=False,
                )
                right_margin = 0.86
            else:
                axes[0].legend(frameon=False)
                right_margin = 0.97
        else:
            right_margin = 0.97
    else:
        right_margin = 0.97

    top_margin = 0.93 if title is not None else 0.97

    fig.tight_layout(
        rect=[0.0, 0.0, right_margin, top_margin]
    )

    savepath = _prepare_output_path(savepath)

    if savepath is not None:
        fig.savefig(
            savepath,
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()

    return fig, axes


def plot_trial_vs_numerical_zero_mode(
    N=51,
    beta=30.0,
    lambda_values=None,
    periodic=True,
    centered=True,
    center_site=0,
    ncols=3,
    savepath=None,
):
    """
    Compare the numerical near-zero mode with the analytical
    trial profile for multiple lambda values.

    The plotted quantity is the probability profile |psi_i|^2.

    Parameters
    ----------
    N : int
        Number of lattice sites. Should be odd.
    beta : float
        Inverse temperature. Only used to obtain the near-zero-mode
        occupation together with the spectral data.
    lambda_values : iterable of float
        Lambda values to display.
    periodic : bool
        Whether to use periodic boundary conditions.
    centered : bool
        If True, shift center_site to the middle of each panel.
    center_site : int
        Site interpreted as the defect center.
    ncols : int
        Number of subplot columns.
    savepath : str or Path, optional
        Figure output path.
    """
    if lambda_values is None:
        lambda_values = [
            -0.75,
            -0.50,
            -0.25,
            0.25,
            0.50,
            0.75,
        ]

    lambda_values = list(lambda_values)

    fig, axes = _make_axes(
        nplots=len(lambda_values),
        ncols=ncols,
        figsize_per_panel=(4.2, 3.2),
        sharex=True,
        sharey=True,
    )

    results = []

    for ax, lam in zip(axes, lambda_values):
        result = compare_trial_and_numerical_mode(
            N=N,
            lam=lam,
            beta=beta,
            periodic=periodic,
        )

        psi = result["psi"]
        phi = result["phi"]

        profile_num = np.abs(psi) ** 2
        profile_trial = np.abs(phi) ** 2

        if centered:
            shift = N // 2 - center_site

            profile_num = np.roll(
                profile_num,
                shift,
            )
            profile_trial = np.roll(
                profile_trial,
                shift,
            )

            x_values = np.arange(N) - N // 2
            xlabel = "Distance from defect"
        else:
            x_values = np.arange(N)
            xlabel = "Site index"

        ax.bar(
            x_values,
            profile_num,
            width=0.8,
            alpha=0.45,
            label="Numerical",
        )

        ax.plot(
            x_values,
            profile_trial,
            marker="o",
            markersize=2.5,
            linewidth=1.3,
            label="Trial",
        )

        ax.set_title(
            rf"$\lambda={lam:.2f}$, "
            rf"$E_0={result['energy']:.1e}$, "
            rf"$P={result['probability_overlap']:.3f}$",
            fontsize=10,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$|\psi_i|^2$")
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.35,
        )

        results.append(result)

    boundary_label = "PBC" if periodic else "OBC"

    _finish_figure(
        fig=fig,
        axes=axes,
        nplots=len(lambda_values),
        title=(
            rf"Numerical and analytical near-zero-mode profiles "
            rf"($N={N}$, {boundary_label})"
        ),
        savepath=savepath,
        legend=True,
        legend_outside=True,
    )

    return results


def plot_correlation_decomposition_vs_lambda(
    N=51,
    beta=30.0,
    lambda_values=None,
    selected_pairs=None,
    periodic=True,
    matrix_element="offdiagonal",
    ncols=2,
    savepath=None,
):
    """
    Plot the exact correlation-matrix-level decomposition

        C_A^defect - C_A^bulk = C_A^zero + R_A.

    Curves shown:
        full correction = C_A^defect - C_A^bulk
        zero-mode term  = C_A^zero
        residual        = R_A

    Parameters
    ----------
    matrix_element : str
        "offdiagonal" -> element C_ij
        "site_i"      -> element C_ii
        "site_j"      -> element C_jj
    """
    if lambda_values is None:
        lambda_values = np.linspace(
            -0.95,
            0.95,
            301,
        )

    lambda_values = np.asarray(
        lambda_values,
        dtype=float,
    )

    if selected_pairs is None:
        selected_pairs = [
            (0, 1),
            (1, 2),
            (12, 13),
            (24, 25),
        ]

    if matrix_element == "offdiagonal":
        row, column = 0, 1
        ylabel = r"Off-diagonal correction"
        element_label = r"$C_{ij}$"
    elif matrix_element == "site_i":
        row, column = 0, 0
        ylabel = r"Occupation correction"
        element_label = r"$C_{ii}$"
    elif matrix_element == "site_j":
        row, column = 1, 1
        ylabel = r"Occupation correction"
        element_label = r"$C_{jj}$"
    else:
        raise ValueError(
            "matrix_element must be one of "
            "'offdiagonal', 'site_i', or 'site_j'."
        )

    fig, axes = _make_axes(
        nplots=len(selected_pairs),
        ncols=ncols,
        figsize_per_panel=(4.8, 3.5),
        sharex=True,
        sharey=False,
    )

    output = {}

    for ax, pair in zip(axes, selected_pairs):
        full_correction = []
        zero_mode_term = []
        residual_term = []
        approximate_value = []
        defect_value = []

        for lam in lambda_values:
            result = two_site_defect_decomposition(
                N=N,
                lam=lam,
                beta=beta,
                pair=pair,
                periodic=periodic,
            )

            delta_full = (
                result["C_defect"]
                - result["C_bulk"]
            )

            full_correction.append(
                np.real_if_close(
                    delta_full[row, column]
                ).real
            )

            zero_mode_term.append(
                np.real_if_close(
                    result["C_zero"][row, column]
                ).real
            )

            residual_term.append(
                np.real_if_close(
                    result["residual"][row, column]
                ).real
            )

            approximate_value.append(
                np.real_if_close(
                    result["C_approx"][row, column]
                ).real
            )

            defect_value.append(
                np.real_if_close(
                    result["C_defect"][row, column]
                ).real
            )

        full_correction = np.asarray(full_correction)
        zero_mode_term = np.asarray(zero_mode_term)
        residual_term = np.asarray(residual_term)
        approximate_value = np.asarray(approximate_value)
        defect_value = np.asarray(defect_value)

        ax.plot(
            lambda_values,
            full_correction,
            linewidth=2.0,
            label=(
                r"$C_A^{\rm defect}"
                r"-C_A^{\rm bulk}$"
            ),
        )

        ax.plot(
            lambda_values,
            zero_mode_term,
            linestyle="--",
            linewidth=1.8,
            label=r"$C_A^{0}$",
        )

        ax.plot(
            lambda_values,
            residual_term,
            linestyle=":",
            linewidth=2.0,
            label=r"$R_A$",
        )

        ax.axhline(
            0.0,
            linewidth=0.8,
        )
        ax.axvline(
            0.0,
            linewidth=0.8,
            linestyle=":",
        )

        ax.set_title(f"Pair {pair}")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(
            rf"{ylabel} for {element_label}"
        )
        ax.grid(alpha=0.3)

        output[pair] = {
            "lambda": lambda_values,
            "full_correction": full_correction,
            "zero_mode": zero_mode_term,
            "residual": residual_term,
            "approximate_value": approximate_value,
            "defect_value": defect_value,
        }

    _finish_figure(
        fig=fig,
        axes=axes,
        nplots=len(selected_pairs),
        title=(
            rf"Correlation-matrix defect correction "
            rf"($N={N}$, $\beta={beta}$)"
        ),
        savepath=savepath,
        legend=True,
        legend_outside=True,
    )

    return output


def plot_correlation_approximation_vs_lambda(
    N=51,
    beta=30.0,
    lambda_values=None,
    selected_pairs=None,
    periodic=True,
    matrix_element="offdiagonal",
    ncols=2,
    savepath=None,
):
    """
    Directly compare

        C_A^approx = C_A^bulk + C_A^zero

    with the full numerical defect correlation matrix.

    This plot answers whether the bulk + near-zero-mode approximation
    reproduces the selected correlation-matrix element.
    """
    if lambda_values is None:
        lambda_values = np.linspace(
            -0.95,
            0.95,
            301,
        )

    lambda_values = np.asarray(
        lambda_values,
        dtype=float,
    )

    if selected_pairs is None:
        selected_pairs = [
            (0, 1),
            (1, 2),
            (12, 13),
            (24, 25),
        ]

    if matrix_element == "offdiagonal":
        row, column = 0, 1
        ylabel = r"$C_{ij}$"
    elif matrix_element == "site_i":
        row, column = 0, 0
        ylabel = r"$C_{ii}$"
    elif matrix_element == "site_j":
        row, column = 1, 1
        ylabel = r"$C_{jj}$"
    else:
        raise ValueError(
            "matrix_element must be one of "
            "'offdiagonal', 'site_i', or 'site_j'."
        )

    fig, axes = _make_axes(
        nplots=len(selected_pairs),
        ncols=ncols,
        figsize_per_panel=(4.8, 3.5),
        sharex=True,
        sharey=False,
    )

    output = {}

    for ax, pair in zip(axes, selected_pairs):
        bulk_values = []
        approximate_values = []
        defect_values = []
        residual_values = []

        for lam in lambda_values:
            result = two_site_defect_decomposition(
                N=N,
                lam=lam,
                beta=beta,
                pair=pair,
                periodic=periodic,
            )

            bulk_values.append(
                np.real_if_close(
                    result["C_bulk"][row, column]
                ).real
            )

            approximate_values.append(
                np.real_if_close(
                    result["C_approx"][row, column]
                ).real
            )

            defect_values.append(
                np.real_if_close(
                    result["C_defect"][row, column]
                ).real
            )

            residual_values.append(
                np.real_if_close(
                    result["residual"][row, column]
                ).real
            )

        bulk_values = np.asarray(bulk_values)
        approximate_values = np.asarray(
            approximate_values
        )
        defect_values = np.asarray(defect_values)
        residual_values = np.asarray(
            residual_values
        )

        ax.plot(
            lambda_values,
            bulk_values,
            linewidth=1.6,
            label=r"$C_A^{\rm bulk}$",
        )

        ax.plot(
            lambda_values,
            approximate_values,
            linestyle="--",
            linewidth=1.8,
            label=r"$C_A^{\rm bulk+0}$",
        )

        ax.plot(
            lambda_values,
            defect_values,
            linewidth=2.0,
            label=r"$C_A^{\rm defect}$",
        )

        ax.axvline(
            0.0,
            linewidth=0.8,
            linestyle=":",
        )

        ax.set_title(f"Pair {pair}")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

        output[pair] = {
            "lambda": lambda_values,
            "bulk": bulk_values,
            "approximation": approximate_values,
            "defect": defect_values,
            "residual": residual_values,
        }

    _finish_figure(
        fig=fig,
        axes=axes,
        nplots=len(selected_pairs),
        title=(
            rf"Testing the near-zero-mode approximation "
            rf"at correlation-matrix level "
            rf"($N={N}$, $\beta={beta}$)"
        ),
        savepath=savepath,
        legend=True,
        legend_outside=True,
    )

    return output


def plot_concurrence_comparison_vs_lambda(
    N=51,
    beta=30.0,
    lambda_values=None,
    selected_pairs=None,
    periodic=True,
    ncols=2,
    savepath=None,
):
    """
    Compare concurrence obtained from:

        1. the uniform bulk correlation matrix,
        2. bulk + analytical near-zero-mode approximation,
        3. the full numerical defect correlation matrix.
    """
    if lambda_values is None:
        lambda_values = np.linspace(
            -0.95,
            0.95,
            301,
        )

    lambda_values = np.asarray(
        lambda_values,
        dtype=float,
    )

    if selected_pairs is None:
        selected_pairs = [
            (0, 1),
            (1, 2),
            (12, 13),
            (24, 25),
        ]

    fig, axes = _make_axes(
        nplots=len(selected_pairs),
        ncols=ncols,
        figsize_per_panel=(4.8, 3.5),
        sharex=True,
        sharey=True,
    )

    output = {}

    for ax, pair in zip(axes, selected_pairs):
        concurrence_bulk = []
        concurrence_approx = []
        concurrence_defect = []
        concurrence_difference = []

        for lam in lambda_values:
            result = compare_pair_concurrences(
                N=N,
                lam=lam,
                beta=beta,
                pair=pair,
                periodic=periodic,
            )

            concurrence_bulk.append(
                result["concurrence_bulk"]
            )

            concurrence_approx.append(
                result["concurrence_approx"]
            )

            concurrence_defect.append(
                result["concurrence_defect"]
            )

            concurrence_difference.append(
                result["concurrence_defect"]
                - result["concurrence_approx"]
            )

        concurrence_bulk = np.asarray(
            concurrence_bulk,
            dtype=float,
        )

        concurrence_approx = np.asarray(
            concurrence_approx,
            dtype=float,
        )

        concurrence_defect = np.asarray(
            concurrence_defect,
            dtype=float,
        )

        concurrence_difference = np.asarray(
            concurrence_difference,
            dtype=float,
        )

        ax.plot(
            lambda_values,
            concurrence_bulk,
            linewidth=1.7,
            label=r"$\mathcal{C}_W^{\rm bulk}$",
        )

        ax.plot(
            lambda_values,
            concurrence_approx,
            linestyle="--",
            linewidth=1.9,
            label=r"$\mathcal{C}_W^{\rm bulk+0}$",
        )

        ax.plot(
            lambda_values,
            concurrence_defect,
            linewidth=2.1,
            label=r"$\mathcal{C}_W^{\rm defect}$",
        )

        ax.axvline(
            0.0,
            linewidth=0.8,
            linestyle=":",
        )

        ax.set_title(f"Pair {pair}")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Concurrence")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)

        output[pair] = {
            "lambda": lambda_values,
            "bulk": concurrence_bulk,
            "approximation": concurrence_approx,
            "defect": concurrence_defect,
            "difference": concurrence_difference,
        }

    _finish_figure(
        fig=fig,
        axes=axes,
        nplots=len(selected_pairs),
        title=(
            rf"Testing the near-zero-mode approximation "
            rf"for concurrence "
            rf"($N={N}$, $\beta={beta}$)"
        ),
        savepath=savepath,
        legend=True,
        legend_outside=True,
    )

    return output


def plot_concurrence_approximation_error_vs_lambda(
    N=51,
    beta=30.0,
    lambda_values=None,
    selected_pairs=None,
    periodic=True,
    absolute=True,
    ncols=2,
    savepath=None,
):
    """
    Plot the discrepancy between full defect concurrence and
    the bulk + near-zero-mode approximation:

        Delta C_W =
            C_W^defect - C_W^(bulk+0).

    This is not a linear concurrence decomposition; it is only
    an approximation error.
    """
    if lambda_values is None:
        lambda_values = np.linspace(
            -0.95,
            0.95,
            301,
        )

    lambda_values = np.asarray(
        lambda_values,
        dtype=float,
    )

    if selected_pairs is None:
        selected_pairs = [
            (0, 1),
            (1, 2),
            (12, 13),
            (24, 25),
        ]

    fig, axes = _make_axes(
        nplots=len(selected_pairs),
        ncols=ncols,
        figsize_per_panel=(4.8, 3.3),
        sharex=True,
        sharey=False,
    )

    output = {}

    for ax, pair in zip(axes, selected_pairs):
        errors = []

        for lam in lambda_values:
            result = compare_pair_concurrences(
                N=N,
                lam=lam,
                beta=beta,
                pair=pair,
                periodic=periodic,
            )

            error = (
                result["concurrence_defect"]
                - result["concurrence_approx"]
            )

            errors.append(
                abs(error) if absolute else error
            )

        errors = np.asarray(
            errors,
            dtype=float,
        )

        ax.plot(
            lambda_values,
            errors,
            linewidth=2.0,
        )

        ax.axhline(
            0.0,
            linewidth=0.8,
        )
        ax.axvline(
            0.0,
            linewidth=0.8,
            linestyle=":",
        )

        ax.set_title(f"Pair {pair}")
        ax.set_xlabel(r"$\lambda$")

        if absolute:
            ax.set_ylabel(
                r"$|\mathcal{C}_W^{\rm defect}"
                r"-\mathcal{C}_W^{\rm bulk+0}|$"
            )
        else:
            ax.set_ylabel(
                r"$\mathcal{C}_W^{\rm defect}"
                r"-\mathcal{C}_W^{\rm bulk+0}$"
            )

        ax.grid(alpha=0.3)

        output[pair] = {
            "lambda": lambda_values,
            "error": errors,
        }

    _finish_figure(
        fig=fig,
        axes=axes,
        nplots=len(selected_pairs),
        title=(
            rf"Near-zero-mode concurrence approximation error "
            rf"($N={N}$, $\beta={beta}$)"
        ),
        savepath=savepath,
        legend=False,
        legend_outside=False,
    )

    return output