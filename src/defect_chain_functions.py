import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.linalg import eigh, sqrtm
import string
from qutip import Qobj, ptrace
from scipy.special import expit
from pathlib import Path

# --- project paths ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

FIG_ROOT = PROJECT_ROOT / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

FIG_DIR = FIG_ROOT / "defect"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def build_ssh_hamiltonian(N, lam, periodic=False, verbose=False):
    """
    Constructs the SSH Hamiltonian with alternating hoppings t1 = 1 - λ, t2 = 1 + λ.

    Parameters:
        N (int): Number of sites
        lam (float): SSH lambda parameter
        periodic (bool): Whether to add periodic boundary condition
        verbose (bool): If True, prints the resulting Hamiltonian

    Returns:conda init powershell

        ndarray: NxN Hamiltonian matrix
    """
    H = np.zeros((N, N), dtype=np.float64)
    t1, t2 = 1 - lam, 1 + lam

    max_i = N if periodic else N - 1
    for i in range(max_i):
        j = (i + 1) % N
        hopping = t1 if i % 2 == 0 else t2
        H[i, j] = H[j, i] = hopping

    if verbose:
        print(f"SSH Hamiltonian (N={N}, λ={lam}, periodic={periodic}):")
        print(H)

    return H

def compute_rho(beta, lam, pair_idx, N=4, periodic=False, verbose=False):
    """
    Computes the reduced 2-site density matrix (RDM) for a thermal state of the SSH model.
    For N=2, it directly constructs the 2x2 correlation matrix and embeds it in the 4x4 basis.

    Parameters:
        beta (float): Inverse temperature.
        lam (float): SSH lambda parameter.
        pair_idx (tuple): Indices (i, j) of the two sites to trace over.
        N (int): Number of sites in the full chain.
        periodic (bool): Whether to use periodic boundary conditions.
        verbose (bool): If True, prints subcorrelation and reduced density matrix.

    Returns:
        rho (4x4 ndarray): The 2-site reduced density matrix.
    """

    if verbose:
        print(f"\n--- compute_rho called ---")
        print(f"beta = {beta}, lambda = {lam}, pair_idx = {pair_idx}, N = {N}")

    if N == 2:
        t1 = 1 - lam
        H_sp = np.array([[0, t1], [t1, 0]])
        eigvals, eigvecs = eigh(H_sp)

        if verbose:
            print("\nSingle-particle Hamiltonian:")
            print(H_sp)
            print("Eigenvalues:", eigvals)

        f = 1 / (np.exp(beta * eigvals) + 1)
        f_diag = np.diag(f)


        C = eigvecs @ f_diag @ eigvecs.T

        if verbose:
            print("Correlation matrix C:")
            print(C)

        rho = np.zeros((4, 4), dtype=complex)
        rho[1, 1] = C[0, 0]
        rho[1, 2] = C[0, 1]
        rho[2, 1] = C[1, 0]
        rho[2, 2] = C[1, 1]

        if verbose:
            print("Reduced density matrix rho:")
            print(rho)

        return rho

    # General case for N > 2
    H = build_ssh_hamiltonian(N, lam, periodic)
    eigvals, eigvecs = eigh(H)

    if beta > 100:
        f_k = expit(-beta * eigvals)
    else:
        x = beta * eigvals
        x = np.clip(x, -700, 700)
        f_k = 1 / (np.exp(x) + 1)


    C = sum(np.outer(eigvecs[:, k], eigvecs[:, k]) * f_k[k] for k in range(N))


    i, j = pair_idx
    C_sub = C[[i, j]][:, [i, j]]

    if verbose:
        print(f"\nSub correlation matrix C_sub (sites {i},{j}):")
        print(C_sub)

    evals_C, U = eigh(C_sub)
    evals_C = np.clip(evals_C, 1e-10, 1 - 1e-10)

    if verbose:
        print("Eigenvalues of C_sub:", evals_C)

    H_red = U @ np.diag(np.log((1 - evals_C) / evals_C)) @ U.T
    h11, h12 = H_red[0, 0], H_red[0, 1]
    h21, h22 = H_red[1, 0], H_red[1, 1]


    H_occ = np.array([
        [0, 0, 0, 0],
        [0, h22, h12, 0],
        [0, h21, h11, 0],
        [0, 0, 0, h11 + h22]
    ])

    evals_occ, evecs_occ = eigh(H_occ)
    exp_neg_evals = np.exp(-evals_occ)
    Z = np.sum(exp_neg_evals)

    rho = evecs_occ @ np.diag(exp_neg_evals / Z) @ evecs_occ.T

    if verbose:
        print("\nFinal reduced density matrix rho:")
        print(rho)

    return rho

def concurrence_general(rho):
    sy = np.array([[0, -1j], [1j, 0]])
    sy_sy = np.kron(sy, sy)
    rho_tilde = sy_sy @ rho.conj() @ sy_sy
    sqrt_rho = sqrtm(rho)
    R = sqrtm(sqrt_rho @ rho_tilde @ sqrt_rho)
    lambdas = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
    return max(0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3])

def plot_selected_thermal_concurrences_vs_lambda(
    N=4, 
    beta_list=None, 
    lambda_vals=None, 
    periodic=False, 
    selected_pairs=None,
    savepath=None
):
    """
    Thermal concurrence vs λ for selected site pairs (works for even/odd N, OBC/PBC).

    Parameters
    ----------
    N : int
    beta_list : list[float]
    lambda_vals : array-like
    periodic : bool
    selected_pairs : list[tuple[int,int]]  # e.g. [(0,1),(1,2),(N-1,0)]
    """
    if savepath is None:
        savepath = FIG_DIR / f"thermal_concurrence_odd_N{N}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)
    
    if beta_list is None:
        beta_list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    if lambda_vals is None:
        lambda_vals = np.linspace(-1, 1, 300)

   
    if selected_pairs is None:
        if periodic and (N % 2 == 1):           # odd-N PBC
            selected_pairs = [(0, 1), (1, 2), (N-1, 0)]
        elif periodic:                           # even-N PBC
            selected_pairs = [(0, 1), (1, 2)]
        else:                                    # OBC
            selected_pairs = [(0, 1), (1, 2), (0, N-1)]

    num = len(selected_pairs)
    fig, axs = plt.subplots(1, num, figsize=(4*num, 4), sharex=True, sharey=True)
    if num == 1:
        axs = [axs]

    for ax, pair in zip(axs, selected_pairs):
        curves = {}
        for beta in beta_list:
            conc_vals = []
            for lam in lambda_vals:
                rho = compute_rho(beta, lam, pair, N=N, periodic=periodic)
                conc_vals.append(concurrence_general(rho))
            ax.plot(lambda_vals, conc_vals, label=f'β={beta}')
        ax.set_title(f"pair {pair}")
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel('Concurrence')
        ax.grid(True)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='β values', loc='center left',
               bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, title_fontsize=10)

    bc_type = "Periodic" if periodic else "Open"
    plt.suptitle(f'Thermal Concurrence vs λ ({bc_type}, N={N})', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])

    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

#Print concurrence value at desired lambda 

# trial codeda plot_concurrence_map_vs_lambda_index kodu var, ancak şu anda kullanılmıyor. İleride eklenebilir.
# 2D heatmap of concurrence between (ref_site, i) vs lambda.
#-----------------------------------------------

def plot_energy_vs_lambda(N=51, lambda_vals=None, periodic=True):
    """
    Plots all SSH single-particle energy eigenvalues as a function of λ.
    Each λ gives N discrete eigenvalues; plotted as E vs λ scatter points.

    Parameters
    ----------
    N : int
        Number of lattice sites
    lambda_vals : array-like
        Range of λ values (default: np.linspace(-1, 1, 201))
    periodic : bool
        Periodic boundary condition flag
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import eigh

    if lambda_vals is None:
        lambda_vals = np.linspace(-1, 1, 201)

    all_E = []

    for lam in lambda_vals:
        H = build_ssh_hamiltonian(N, lam, periodic)
        E, _ = eigh(H)
        for e in E:
            all_E.append((lam, e))

    all_E = np.array(all_E)
    plt.figure(figsize=(7,5))
    #plt.plot(all_E[:,0], all_E[:,1], s=5, alpha=0.7, color='steelblue')
    plt.scatter(all_E[:, 0], all_E[:, 1], s=5, alpha=0.7, color='steelblue')
    plt.axhline(0, color='gray', ls='--', lw=0.8)
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Energy eigenvalues $E$')
    bc = "PBC" if periodic else "OBC"
    plt.title(fr'SSH energy spectrum $E(\lambda)$  (N={N}, {bc})')
    plt.tight_layout()
    plt.show()

def plot_energy_vs_lambda_multiN(N_list, lambda_vals=None, periodic=True, savepath=None):
    """
    Plots SSH energy spectra for multiple system sizes N
    in a 2x3 subplot grid.

    Parameters
    ----------
    N_list : list or array-like
        List of system sizes (ideally 6 values for a 2x3 grid,
        e.g. [21, 31, 41, 51, 61, 71])
    lambda_vals : array-like, optional
        Range of lambda values (default: np.linspace(-1, 1, 201))
    periodic : bool, optional
        Periodic boundary condition flag
        True  -> PBC
        False -> OBC
    """
   

    if savepath is None:
        savepath = FIG_DIR / f"energy_vs_lambda_multiN.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    if lambda_vals is None:
        lambda_vals = np.linspace(-1, 1, 201)

    n_plots = len(N_list)

    # 2x3 subplot grid
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14, 8),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    for idx, N in enumerate(N_list):
        ax = axes[idx]

        E_vals = []

        for lam in lambda_vals:
            H = build_ssh_hamiltonian(N, lam, periodic)
            E, _ = eigh(H)
            E_vals.append(E)

        E_vals = np.array(E_vals)  # shape = (len(lambda_vals), N)

        # Plot each eigenvalue branch as a function of lambda
        for i in range(E_vals.shape[1]):
            ax.plot(lambda_vals, E_vals[:, i], lw=0.6, color='black')

        ax.axhline(0, color='gray', ls='--', lw=0.8)

        bc = "PBC" if periodic else "OBC"
        ax.set_title(fr"$N={N}$ ({bc})", fontsize=11)
        ax.set_ylabel(r"$E$")

        # İstersen zero-mode bölgesini hafifçe vurgula:
        # ax.axhspan(-0.1, 0.1, color='red', alpha=0.08)

        # İstersen enerji aralığını sabitle:
        # ax.set_ylim(-2.5, 2.5)

    # Eğer 6'dan az subplot varsa boş kalanları kapat
    for j in range(n_plots, nrows * ncols):
        axes[j].axis("off")

    # Alt sıradaki panellere x-label koy
    for ax in axes[-ncols:]:
        ax.set_xlabel(r"$\lambda$")

    fig.suptitle("SSH Energy Spectrum vs $\\lambda$", fontsize=14)
    plt.tight_layout()

    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

####MULTI N with colorful E=0

def plot_energy_vs_lambda_multiN_red(
    N_list,
    lambda_vals=None,
    periodic=True,
    save_path=None,
    highlight_zero_mode=True,
    shade_topological=False,
    zero_tol=1e-10
):
    """
    Plot SSH single-particle energy spectra for multiple system sizes N
    in a 2x3 subplot grid.

    Parameters
    ----------
    N_list : list
        List of system sizes, ideally 6 values for a 2x3 grid.
    lambda_vals : array-like, optional
        Lambda grid. Default: np.linspace(-1, 1, 201)
    periodic : bool, optional
        True  -> PBC
        False -> OBC
    save_path : str or Path, optional
        Output file path.
    highlight_zero_mode : bool, optional
        If True, exact zero-energy branch is highlighted when present.
    shade_topological : bool, optional
        If True, shades the topological region.
        IMPORTANT: here we assume the user's convention:
            topological phase <=> lambda > 0
    zero_tol : float, optional
        Numerical tolerance for identifying zero mode.
    """

    # Default save path
    if save_path is None:
        save_path = FIG_DIR / "energy_vs_lambda_multiN.pdf"
    else:
        save_path = Path(save_path).resolve()

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Default lambda grid
    if lambda_vals is None:
        lambda_vals = np.linspace(-1.0, 1.0, 201)

    n_plots = len(N_list)
    nrows, ncols = 2, 3

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14, 8),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    for idx, N in enumerate(N_list):
        ax = axes[idx]

        E_vals = []

        for lam in lambda_vals:
            H = build_ssh_hamiltonian(N, lam, periodic)
            E, _ = eigh(H)
            E_vals.append(E)

        E_vals = np.array(E_vals)   # shape = (len(lambda_vals), N)

        # Optional: shade topological region
        # User convention: topological for lambda > 0
        if shade_topological:
            ax.axvspan(0.0, lambda_vals.max(), alpha=0.08, color='orange', zorder=0)

        # Plot each eigenvalue branch
        for i in range(E_vals.shape[1]):
            branch = E_vals[:, i]

            is_exact_zero_branch = np.all(np.abs(branch) < zero_tol)

            # Highlight only if requested, and especially relevant for odd-N OBC
            if (
                highlight_zero_mode
                and (not periodic)
                and (N % 2 == 1)
                and is_exact_zero_branch
            ):
                ax.plot(
                    lambda_vals, branch,
                    lw=2.2,
                    color='crimson',
                    zorder=5,
                    label='zero mode' if idx == 0 else None
                )
            else:
                ax.plot(
                    lambda_vals, branch,
                    lw=0.6,
                    color='black',
                    alpha=0.95,
                    zorder=2
                )

        # E=0 reference line
        ax.axhline(0.0, color='gray', ls='--', lw=0.7, alpha=0.5, zorder=1)

        bc = "PBC" if periodic else "OBC"
        ax.set_title(fr"$N={N}$ ({bc})", fontsize=12)
        ax.set_ylabel(r"$E$")

    # Turn off unused panels
    for j in range(n_plots, nrows * ncols):
        axes[j].axis("off")

    # x-label only on bottom row
    for ax in axes[-ncols:]:
        ax.set_xlabel(r"$\lambda$")

    # Legend only if zero mode was labeled
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(r"SSH Energy Spectrum vs $\lambda$", fontsize=16)
    plt.tight_layout()

    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

def plot_energy_vs_lambda_multiN_blue(
    N_list,
    lambda_vals=None,
    periodic=True,
    save_path=None,
    highlight_near_zero=True,
    near_zero_tol=0.08,
    shade_topological=False
):
    """
    SSH energy spectrum vs lambda for multiple N.

    Highlights near-zero edge modes (in topological region λ > 0) in BLUE.

    Convention:
        topological phase <=> lambda > 0
    """

    # Save path
    if save_path is None:
        save_path = FIG_DIR / "energy_vs_lambda_multiN_blue.pdf"
    else:
        save_path = Path(save_path).resolve()

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Lambda grid
    if lambda_vals is None:
        lambda_vals = np.linspace(-1.0, 1.0, 201)

    topo_mask = lambda_vals > 0

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14, 8),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    legend_added = False

    for idx, N in enumerate(N_list):
        ax = axes[idx]

        E_vals = []

        for lam in lambda_vals:
            H = build_ssh_hamiltonian(N, lam, periodic)
            E, _ = eigh(H)
            E_vals.append(E)

        E_vals = np.array(E_vals)

        # Optional: shade topological region
        if shade_topological:
            ax.axvspan(0.0, lambda_vals.max(), alpha=0.08, color='orange', zorder=0)

        # Plot branches
        for i in range(E_vals.shape[1]):
            branch = E_vals[:, i]

            # minimum |E| in topological region
            if np.any(topo_mask):
                min_abs_topo = np.min(np.abs(branch[topo_mask]))
            else:
                min_abs_topo = np.inf

            is_near_zero = (
                highlight_near_zero
                and (not periodic)   # only meaningful for OBC
                and (min_abs_topo < near_zero_tol)
            )

            if is_near_zero:
                ax.plot(
                    lambda_vals, branch,
                    lw=1.8,
                    color='royalblue',
                    zorder=4,
                    label='edge mode' if not legend_added else None
                )
                legend_added = True
            else:
                ax.plot(
                    lambda_vals, branch,
                    lw=0.6,
                    color='black',
                    alpha=0.95,
                    zorder=2
                )

        # E=0 reference
        ax.axhline(0, color='gray', ls='--', lw=0.7, alpha=0.5)

        bc = "PBC" if periodic else "OBC"
        ax.set_title(fr"$N={N}$ ({bc})", fontsize=12)
        ax.set_ylabel(r"$E$")

    # turn off unused axes
    for j in range(len(N_list), nrows * ncols):
        axes[j].axis("off")

    # x labels
    for ax in axes[-ncols:]:
        ax.set_xlabel(r"$\lambda$")

    # legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(r"SSH Energy Spectrum vs $\lambda$", fontsize=16)
    plt.tight_layout()

    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

## min energy
def plot_min_abs_energy_vs_lambda(
    N=51,
    lambda_vals=None,
    periodic=True,
    plot_abs=True,
    log10y=False,
    eps_floor=1e-16,
    zero_tol=1e-12,
    show_zero_tol=True,
    return_data=True
):
    """
    For each lambda, diagonalize the SSH Hamiltonian and select the eigenvalue
    closest to zero.

    Parameters
    ----------
    plot_abs : bool
        True  -> plot |E|min(lambda)
        False -> plot signed E*(lambda), where E* is the eigenvalue with minimal |E|.
    log10y : bool
        If True, plot log10(|E|min).
    eps_floor : float
        Floor used when taking log10.
    zero_tol : float
        Numerical threshold below which the mode is treated as "exact zero".
    show_zero_tol : bool
        If True, draw a horizontal guide for zero_tol on the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import eigh

    if lambda_vals is None:
        lambda_vals = np.linspace(-1.0, 1.0, 401)

    Emin = np.zeros(len(lambda_vals), dtype=float)
    E_signed = np.zeros(len(lambda_vals), dtype=float)
    idx_min = np.zeros(len(lambda_vals), dtype=int)

    for m, lam in enumerate(lambda_vals):
        H = build_ssh_hamiltonian(N, lam, periodic)
        E, _ = eigh(H)   # sorted ascending
        k = int(np.argmin(np.abs(E)))
        idx_min[m] = k
        E_signed[m] = float(E[k])
        Emin[m] = float(np.abs(E[k]))

    is_exact_zero = Emin < zero_tol

    if log10y:
        y_plot = np.log10(np.maximum(Emin, eps_floor))
        y_label = r'$\log_{10}(|E|_{\min})$'
    else:
        y_plot = Emin if plot_abs else E_signed
        y_label = r'$|E|_{\min}$' if plot_abs else r'$E^*(\lambda)$'

    plt.figure(figsize=(7, 4))
    plt.plot(lambda_vals, y_plot, linestyle='-', lw=1.5)

    if not log10y:
        plt.axhline(0, color='gray', ls='--', lw=0.8)

    if show_zero_tol and plot_abs:
        if log10y:
            plt.axhline(np.log10(zero_tol), color='red', ls=':', lw=1.0,
                        label=fr'zero tol = {zero_tol:.0e}')
        else:
            plt.axhline(zero_tol, color='red', ls=':', lw=1.0,
                        label=fr'zero tol = {zero_tol:.0e}')

    # exact-zero points
    if np.any(is_exact_zero):
        if log10y:
            plt.scatter(
                lambda_vals[is_exact_zero],
                np.log10(np.maximum(Emin[is_exact_zero], eps_floor)),
                s=12, zorder=3, label='exact-zero region'
            )
        elif plot_abs:
            plt.scatter(
                lambda_vals[is_exact_zero],
                Emin[is_exact_zero],
                s=12, zorder=3, label='exact-zero region'
            )
        else:
            plt.scatter(
                lambda_vals[is_exact_zero],
                E_signed[is_exact_zero],
                s=12, zorder=3, label='exact-zero region'
            )

    plt.xlabel(r'$\lambda$')
    plt.ylabel(y_label)
    bc = "PBC" if periodic else "OBC"
    plt.title(fr'Closest-to-zero energy vs $\lambda$ (N={N}, {bc})')
    plt.grid(True, ls='--', alpha=0.4)

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    if return_data:
        return lambda_vals, Emin, E_signed, idx_min, is_exact_zero
        
#subplot min abs energy for multiple N values
def plot_min_abs_energy_multiN(
    N_list,
    lambda_vals=None,
    periodic=True,
    log10y=False,
    zero_tol=1e-12,
    eps_floor=1e-16,
    ncols=None,
    figsize_per_plot=(5, 3.5),
    sharey=True,
    savepath=None
):
    """
    Plot |E|min(λ) for multiple system sizes N in a grid of subplots.

    Parameters
    ----------
    N_list : list
        List of system sizes
    lambda_vals : array-like
        Lambda grid
    periodic : bool
        PBC or OBC
    log10y : bool
        Use log10 scale
    zero_tol : float
        Threshold for "exact zero"
    ncols : int or None
        Number of columns (default: all in one row)
    figsize_per_plot : tuple
        Size per subplot
    sharey : bool
        Share y-axis across plots
    """

    if savepath is None:
        savepath = FIG_DIR / f"zero_energy_vs_lambda_multiN.pdf"
    else:
        savepath = Path(savepath).resolve()

    if lambda_vals is None:
        lambda_vals = np.linspace(-1.0, 1.0, 401)

    n_plots = len(N_list)

    # layout ayarı
    if ncols is None:
        ncols = n_plots
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows),
        sharey=sharey
    )

    # axes flatten (tek subplot olsa bile)
    axes = np.atleast_1d(axes).flatten()

    for ax, N in zip(axes, N_list):

        Emin = []

        for lam in lambda_vals:
            H = build_ssh_hamiltonian(N, lam, periodic)
            E, _ = eigh(H)
            Emin.append(np.min(np.abs(E)))

        Emin = np.array(Emin)

        if log10y:
            y = np.log10(np.maximum(Emin, eps_floor))
            ylabel = r'$\log_{10}(|E|_{\min})$'
        else:
            y = Emin
            ylabel = r'$|E|_{\min}$'

        ax.plot(lambda_vals, y, lw=1.5)

        # zero tolerance çizgisi
        if log10y:
            ax.axhline(np.log10(zero_tol), color='red', ls=':', lw=1)
        else:
            ax.axhline(zero_tol, color='red', ls=':', lw=1)

        # kritik nokta
        ax.axvline(0, color='gray', ls='--', alpha=0.5)

        bc = "PBC" if periodic else "OBC"
        ax.set_title(fr"$N={N}$ ({bc})")
        ax.set_xlabel(r'$\lambda$')
        ax.grid(True, ls='--', alpha=0.4)

    # boş subplotları kapat
    for j in range(len(N_list), len(axes)):
        axes[j].axis("off")

    # sadece sol altta ylabel
    axes[0].set_ylabel(ylabel)

    plt.suptitle(r'Closest-to-zero energy vs $\lambda$ (finite-size scaling)', fontsize=14)
    plt.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()


### Energy versus lambda distribution plots in terms of mode index (sorted by energy) rather than lambda.

def plot_spectrum(
    N=51,
    lam=0.6,
    periodic=True,
    highlight_zero=True,
    save_path=None
):
    """
    SSH single-particle energy spectrum: mode index vs energy.

    Parameters
    ----------
    highlight_zero : bool
        If True, highlights the eigenvalue with smallest |E|.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    """

    H = build_ssh_hamiltonian(N, lam, periodic)
    E, U = eigh(H)
    x = np.arange(len(E))

    # Save path handling
    if save_path is None:
        save_path = FIG_DIR / f"spectrum_N{N}_lam{lam:.2f}_{'PBC' if periodic else 'OBC'}.pdf"
    else:
        save_path = Path(save_path).resolve()

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot
    plt.figure(figsize=(7,4))
    plt.plot(x, E, 'o', ms=4, alpha=0.85)

    plt.axhline(0, ls='--', c='gray', lw=0.8)

    if highlight_zero:
        zi = int(np.argmin(np.abs(E)))
        plt.plot([zi], [E[zi]], 'o', ms=8, color='crimson')

        plt.annotate(
            f"min|E| ≈ {E[zi]:.2e}",
            (zi, E[zi]),
            xytext=(zi+1, E[zi]),
            arrowprops=dict(arrowstyle='->', lw=0.8)
        )

    bc = "PBC" if periodic else "OBC"
    plt.title(f"SSH Spectrum (N={N}, λ={lam}, {bc})")
    plt.xlabel("Mode index")
    plt.ylabel("Energy")

    plt.tight_layout()

    # Save
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()

def plot_spectrum_multiN_vertical(
    N_list,
    lam=0.0,
    periodic=True,
    highlight_zero=True,
    save_path=None,
    figsize_per_panel=(7, 2.8),
    sharey=True
):
    """
    Plot SSH single-particle spectra for multiple system sizes N
    at fixed lambda, stacked vertically.

    Parameters
    ----------
    N_list : list
        List of system sizes.
    lam : float
        Fixed lambda value.
    periodic : bool
        True -> PBC, False -> OBC
    highlight_zero : bool
        If True, highlight the eigenvalue with smallest |E|.
    save_path : str or Path, optional
        Output path for saving figure.
    figsize_per_panel : tuple
        Size per subplot panel, e.g. (width, height).
    sharey : bool
        Whether to share the y-axis across panels.
    """

    if save_path is None:
        bc = "PBC" if periodic else "OBC"
        save_path = FIG_DIR / f"spectrum_multiN_lam{lam:.2f}_{bc}_vertical.pdf"
    else:
        save_path = Path(save_path).resolve()

    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_plots = len(N_list)

    fig, axes = plt.subplots(
        n_plots, 1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * n_plots),
        sharey=sharey
    )

    axes = np.atleast_1d(axes)

    for ax, N in zip(axes, N_list):
        H = build_ssh_hamiltonian(N, lam, periodic)
        E, U = eigh(H)
        x = np.arange(len(E))

        ax.plot(x, E, 'o', ms=3.5, alpha=0.85)
        ax.axhline(0, ls='--', c='gray', lw=0.8)

        if highlight_zero:
            zi = int(np.argmin(np.abs(E)))
            ax.plot([zi], [E[zi]], 'o', ms=7, color='crimson')
            ax.annotate(
                f"min|E| ≈ {E[zi]:.2e}",
                (zi, E[zi]),
                xytext=(zi + max(1, len(E)//40), E[zi]),
                arrowprops=dict(arrowstyle='->', lw=0.8),
                fontsize=9
            )

        bc = "PBC" if periodic else "OBC"
        ax.set_title(fr"$N={N}$, $\lambda={lam}$ ({bc})", fontsize=11)
        ax.set_ylabel("Energy")
        ax.grid(axis='y', ls='--', alpha=0.35)

    axes[-1].set_xlabel("Mode index")

    fig.suptitle(fr"SSH Spectrum for different $N$ at fixed $\lambda={lam}$", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

### Zero mode distribution

