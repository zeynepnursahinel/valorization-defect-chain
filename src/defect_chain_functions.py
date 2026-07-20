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
#-----------------------------------------------

#Energy in terms of MODE INDEX (sorted by energy) rather than lambda.

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
#-----------------------------------------------

#Analyzing Exact zero energy whether exists or not, NUMERİCAL ZERO OR ANALYTICAL ZERO

def analyze_zero_mode_vs_lambda(
    N=51,
    lambda_vals=None,
    periodic=True,
    tol_eig=1e-14,
    tol_res=1e-12,
    plot=True,
    log10y=False,
    eps_floor=1e-16
):
    """
    Analyze closest-to-zero eigenvalue and check whether it is an exact zero mode.

    Criteria:
        - |E| < tol_eig AND residual < tol_res  -> numerical exact zero
        - otherwise -> near-zero mode
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import eigh

    if lambda_vals is None:
        lambda_vals = np.linspace(-1.0, 1.0, 401)

    Emin = []
    residuals = []
    is_numerical_zero = []

    for lam in lambda_vals:
        H = build_ssh_hamiltonian(N, lam, periodic)
        E, V = eigh(H)

        k = int(np.argmin(np.abs(E)))
        e_min = E[k]
        phi = V[:, k]

        res = np.linalg.norm(H @ phi)

        Emin.append(abs(e_min))
        residuals.append(res)

        is_numerical_zero.append(
            (abs(e_min) < tol_eig) and (res < tol_res)
        )

    Emin = np.array(Emin)
    residuals = np.array(residuals)
    is_numerical_zero = np.array(is_numerical_zero)

    # ---------------- PLOT ----------------
    if plot:
        plt.figure(figsize=(7, 4))

        if log10y:
            y_plot = np.log10(np.maximum(Emin, eps_floor))
            plt.ylabel(r'$\log_{10}(|E|_{\min})$')
        else:
            y_plot = Emin
            plt.ylabel(r'$|E|_{\min}$')

        plt.plot(lambda_vals, y_plot, lw=1.5, label='closest-to-zero mode')

        # tolerance lines
        if log10y:
            plt.axhline(np.log10(tol_eig), color='red', ls=':', label='eig tol')
        else:
            plt.axhline(tol_eig, color='red', ls=':', label='eig tol')

        # numerical zero points
        if np.any(is_numerical_zero):
            if log10y:
                plt.scatter(
                    lambda_vals[is_numerical_zero],
                    np.log10(np.maximum(Emin[is_numerical_zero], eps_floor)),
                    s=15,
                    label='numerical zero'
                )
            else:
                plt.scatter(
                    lambda_vals[is_numerical_zero],
                    Emin[is_numerical_zero],
                    s=15,
                    label='numerical zero'
                )

        plt.xlabel(r'$\lambda$')
        bc = "PBC" if periodic else "OBC"
        plt.title(f'Near-zero vs numerical-zero modes (N={N}, {bc})')

        plt.grid(True, ls='--', alpha=0.4)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    return lambda_vals, Emin, residuals, is_numerical_zero
###########################################################################
## min energy AMA BU FONKSİYONLARDA HATA VAR BENCE
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
###########################################################################
#-----------------------------------------------

def plot_wavefunction_profile(N=51, lam=0.6, periodic=True, mode_index='auto'):
    """
    Seçilen modun uzaysal profilini çizer: |ψ(i)|^2.
    mode_index='auto' → |E| en küçük mod.
    """
    H = build_ssh_hamiltonian(N, lam, periodic)
    E, U = eigh(H)
    if mode_index == 'auto':
        mode_index = int(np.argmin(np.abs(E)))
    psi = U[:, mode_index]
    prof = np.abs(psi)**2
    sites = np.arange(N)

    plt.figure(figsize=(7,3.5))
    plt.bar(sites, prof, width=0.8, alpha=0.9)
    plt.xlabel("Site index i")
    plt.ylabel(r"$|\psi(i)|^2$")
    bc = "PBC" if periodic else "OBC"
    plt.title(f"Mode {mode_index}, E ≈ {E[mode_index]:.2e}  (N={N}, λ={lam}, {bc})")
    plt.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return mode_index, E[mode_index], prof

def trial_near_zero_profile_old(N=51, lam=0.6, normalize=True):
    """
    Odd-N PBC SSH chain için approximate near-zero defect-mode profile.

    Uses:
        r = t1/t2 = (1-lambda)/(1+lambda)
        phi_{2n}   = (-r)^n
        phi_{2n+1} = (-1)^{n+1} r^{M-n}

    valid mainly for odd N and |r| < 1.
    """
    import numpy as np

    if N % 2 == 0:
        raise ValueError("This trial profile is intended for odd N.")

    M = (N - 1) // 2
    t1 = 1 - lam
    t2 = 1 + lam
    r = t1 / t2

    phi = np.zeros(N, dtype=float)

    # even sites: j = 2n, n = 0,...,M
    for n in range(M + 1):
        phi[2*n] = (-r)**n

    # odd sites: j = 2n+1, n = 0,...,M-1
    for n in range(M):
        phi[2*n + 1] = ((-1)**(n + 1)) * (r**(M - n))

    if normalize:
        phi = phi / np.linalg.norm(phi)

    return phi

def trial_near_zero_profile(N=51, lam=0.6, normalize=True):
    """
    Odd-N PBC SSH chain için piecewise approximate near-zero defect-mode profile.

    For lambda > 0:
        r = t1/t2
        phi_{2n}   = (-r)^n
        phi_{2n+1} = (-1)^{n+1} r^{M-n}

    For lambda < 0:
        s = t2/t1 = 1/r
        phi_{2n}   = (-1)^n s^{M-n}
        phi_{2n+1} = (-1)^{n+1} s^n

    Valid mainly for odd N away from lambda = 0.
    """
    import numpy as np

    if N % 2 == 0:
        raise ValueError("This trial profile is intended for odd N.")

    M = (N - 1) // 2
    t1 = 1 - lam
    t2 = 1 + lam

    phi = np.zeros(N, dtype=float)

    if lam > 0:
        r = t1 / t2

        for n in range(M + 1):
            phi[2*n] = (-r)**n

        for n in range(M):
            phi[2*n + 1] = (-1)**(n + 1) * r**(M - n)

    elif lam < 0:
        s = t2 / t1

        for n in range(M + 1):
            phi[2*n] = (-1)**n * s**(M - n)

        for n in range(M):
            phi[2*n + 1] = (-1)**(n + 1) * s**n

    else:
        # lambda = 0: localized ansatz is not meaningful
        phi[:] = 1.0

    if normalize:
        norm = np.linalg.norm(phi)
        if norm > 0:
            phi = phi / norm

    return phi

def compare_numeric_and_trial_profile(N=51, lam=0.6, periodic=True, mode_index="auto"):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import eigh

    H = build_ssh_hamiltonian(N, lam, periodic)
    E, U = eigh(H)

    if mode_index == "auto":
        mode_index = int(np.argmin(np.abs(E)))

    psi_num = U[:, mode_index]
    psi_trial = trial_near_zero_profile(N=N, lam=lam, normalize=True)

    # Eigenvectors have arbitrary overall sign, so align signs
    if np.dot(psi_num, psi_trial) < 0:
        psi_trial = -psi_trial

    overlap = abs(np.dot(psi_num, psi_trial))**2

    sites = np.arange(N)

    plt.figure(figsize=(7, 3.5))
    plt.bar(sites - 0.18, np.abs(psi_num)**2, width=0.35, label="numeric")
    plt.bar(sites + 0.18, np.abs(psi_trial)**2, width=0.35, alpha=0.7, label="trial")

    plt.xlabel("Site index i")
    plt.ylabel(r"$|\psi(i)|^2$")
    plt.title(
        fr"N={N}, $\lambda$={lam}, E={E[mode_index]:.2e}, overlap={overlap:.4f}"
    )
    plt.grid(axis="y", ls="--", alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return mode_index, E[mode_index], overlap, psi_num, psi_trial


##--------------------------------------------


def plot_numeric_wavefunction_profile(
    N=51,
    lam=0.6,
    periodic=True,
    mode_index="auto",
    centered=True,
    center_site=0,
    savepath=None,
    return_data=True
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import eigh
    from pathlib import Path

    H = build_ssh_hamiltonian(N, lam, periodic=periodic)
    E, U = eigh(H)

    if mode_index == "auto":
        mode_index = int(np.argmin(np.abs(E)))

    psi = U[:, mode_index]
    psi = psi / np.linalg.norm(psi)

    prof = np.abs(psi)**2

    if centered:
        # Put center_site at the middle of the plot
        shift = N // 2 - center_site
        prof_plot = np.roll(prof, shift)

        x = np.arange(N) - N // 2
        xlabel = "Distance from defect"
        title_extra = "centered"
    else:
        prof_plot = prof
        x = np.arange(N)
        xlabel = "Site index $i$"
        title_extra = "site basis"

    plt.figure(figsize=(7, 3.5))
    plt.bar(x, prof_plot, width=0.8, alpha=0.9)

    plt.xlabel(xlabel)
    plt.ylabel(r"$|\psi_i|^2$")

    bc = "PBC" if periodic else "OBC"
    plt.title(
        fr"Mode {mode_index}, $E={E[mode_index]:.2e}$ "
        fr"$(N={N}, \lambda={lam}, {bc}, {title_extra})$"
    )

    plt.grid(axis="y", ls="--", alpha=0.5)
    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight")

    plt.show()

    if return_data:
        return {
            "mode_index": mode_index,
            "energy": E[mode_index],
            "psi": psi,
            "profile": prof,
            "x_plot": x,
            "profile_plot": prof_plot,
            "eigenvalues": E
        }

def plot_centered_profiles_multi_lambda(
    N=51,
    lambda_list=(0.1, 0.3, 0.5, 0.8),
    periodic=True,
    center_site=0,
    savepath=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import eigh
    from pathlib import Path
    import math

    n_plots = len(lambda_list)

    # 🔥 grid boyutu otomatik
    ncols = int(np.ceil(np.sqrt(n_plots)))
    nrows = int(np.ceil(n_plots / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True, sharey=True)
    axs = np.array(axs).ravel()

    for i, lam in enumerate(lambda_list):
        ax = axs[i]

        H = build_ssh_hamiltonian(N, lam, periodic=periodic)
        E, U = eigh(H)

        mode_index = int(np.argmin(np.abs(E)))
        psi = U[:, mode_index]
        psi = psi / np.linalg.norm(psi)
        prof = np.abs(psi)**2

        shift = N // 2 - center_site
        prof_shifted = np.roll(prof, shift)
        x = np.arange(N) - N // 2

        ax.bar(x, prof_shifted, width=0.8, alpha=0.9)
        ax.set_title(fr"$\lambda={lam}$, $E={E[mode_index]:.2e}$", fontsize=10)
        ax.grid(axis="y", ls="--", alpha=0.4)

    # 🔥 fazla subplotları kapat
    for j in range(n_plots, len(axs)):
        axs[j].axis("off")

    # eksen label
    for ax in axs[:n_plots]:
        ax.set_xlabel("Distance from defect")
        ax.set_ylabel(r"$|\psi_i|^2$")

    fig.suptitle(fr"Near-zero mode profiles ($N={N}$, PBC, centered)", y=1.02)
    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight", dpi=300)

    plt.show()

def compare_numeric_and_trial_multi_lambda(
    N=51,
    lambda_list=(0.25, 0.5, 0.75, 1.0),
    periodic=True,
    centered=True,
    center_site=0,
    savepath=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import eigh
    from pathlib import Path

    n_plots = len(lambda_list)
    ncols = int(np.ceil(np.sqrt(n_plots)))
    nrows = int(np.ceil(n_plots / ncols))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4.2*ncols, 3.2*nrows),
        sharex=True,
        sharey=True
    )
    axs = np.array(axs).ravel()

    results = []

    for i, lam in enumerate(lambda_list):
        ax = axs[i]

        H = build_ssh_hamiltonian(N, lam, periodic=periodic)
        E, U = eigh(H)

        mode_index = int(np.argmin(np.abs(E)))
        psi_num = U[:, mode_index]
        psi_num = psi_num / np.linalg.norm(psi_num)

        psi_trial = trial_near_zero_profile(
            N=N,
            lam=lam,
            normalize=True
        )

        # align global sign
        if np.dot(psi_num, psi_trial) < 0:
            psi_trial = -psi_trial

        wave_overlap = abs(np.dot(psi_num, psi_trial))**2

        prof_num = np.abs(psi_num)**2
        prof_trial = np.abs(psi_trial)**2

        # probability-profile overlap, insensitive to signs
        prob_overlap = (np.sum(np.sqrt(prof_num * prof_trial)))**2

        if centered:
            shift = N // 2 - center_site
            x = np.arange(N) - N // 2
            prof_num_plot = np.roll(prof_num, shift)
            prof_trial_plot = np.roll(prof_trial, shift)
            xlabel = "Distance from defect"
        else:
            x = np.arange(N)
            prof_num_plot = prof_num
            prof_trial_plot = prof_trial
            xlabel = "Site index $i$"

        ax.bar(
            x,
            prof_num_plot,
            width=0.8,
            alpha=0.45,
            label="numeric"
        )

        ax.plot(
            x,
            prof_trial_plot,
            marker="o",
            markersize=2.5,
            lw=1.2,
            label="trial"
        )

        ax.set_title(
            fr"$\lambda={lam}$, $E={E[mode_index]:.1e}$, "
            fr"$P={prob_overlap:.3f}$",
            fontsize=10
        )

        ax.grid(axis="y", ls="--", alpha=0.35)

        results.append({
            "lambda": lam,
            "mode_index": mode_index,
            "energy": E[mode_index],
            "wave_overlap": wave_overlap,
            "probability_overlap": prob_overlap
        })

    for j in range(n_plots, len(axs)):
        axs[j].axis("off")

    for ax in axs[:n_plots]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$|\psi_i|^2$")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right")

    fig.suptitle(
        fr"Numerical vs trial near-zero profiles ($N={N}$, PBC, centered)",
        y=1.02
    )

    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight", dpi=300)

    plt.show()

    return results

##--------------------------------------------

##CONCURRENCE

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

    ncols = 2
    nrows = int(np.ceil(num / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True, sharey=True)
    axs = np.array(axs).reshape(-1)

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

def plot_concurrence_compare_N(
    N_list=(50, 51),
    beta=30.0,
    lambda_vals=None,
    periodic=True,
    selected_pairs=None,
    savepath=None
):
    """
    Aynı pair için farklı N değerlerini (ör. 50 vs 51) aynı ax üzerinde karşılaştırmalı plotlar.

    N_list        : karşılaştırmak istediğin site sayıları (örn. (50,51))
    beta          : tek bir β değeri
    lambda_vals   : λ grid'i (default: np.linspace(-1,1,301))
    periodic      : PBC/OBC
    selected_pairs: [(i,j), ...] şeklinde site çiftleri
    """
    if savepath is None:
        savepath = FIG_DIR / f"thermal_concurrence_comparence.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    if lambda_vals is None:
        lambda_vals = np.linspace(-1, 1, 301)

    if selected_pairs is None:
        raise ValueError("selected_pairs bir liste olarak verilmeli, örn: [(24,25), (25,26)]")

    num = len(selected_pairs)
    ncols = 2
    nrows = int(np.ceil(num / ncols))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 3.2*nrows),
        sharex=True,
        sharey=True
    )

    axs = np.array(axs).reshape(-1)

    for ax, pair in zip(axs, selected_pairs):
        for N in N_list:
            conc_vals = []
            for lam in lambda_vals:
                rho = compute_rho(beta, lam, pair, N=N, periodic=periodic)
                conc_vals.append(concurrence_general(rho))
            ax.plot(lambda_vals, conc_vals, label=f'N={N}')

        ax.set_title(f"pair {pair}")
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel('Concurrence')
        ax.grid(True)

    bc_type = "Periodic" if periodic else "Open"
    fig.suptitle(fr'Thermal Concurrence vs $\lambda$ (β={beta}, {bc_type})', fontsize=14)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='System size', loc='center left',
               bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, title_fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    plt.show()

    fig.savefig(savepath, bbox_inches="tight")

def find_lambda_crit_for_pair(pair, N, beta, lambda_vals, periodic=True, eps=1e-4):
    """
    Belirli bir (i,j) pair'i için concurrence'ın 'entangled ↔ separable' 
    geçtiği λ_crit değerini bulur.

    Yöntem: conc(λ) > eps / < eps maske değiştiği noktayı bulup,
    en |λ|'ye yakın geçiş noktasını seçiyoruz (topolojik geçiş civarı).
    """
    i, j = pair
    conc_vals = []
    for lam in lambda_vals:
        rho = compute_rho(beta, lam, pair, N=N, periodic=periodic)
        conc_vals.append(concurrence_general(rho))
    conc_vals = np.array(conc_vals)

    mask = conc_vals > eps
    # maske nerede değişiyor? (True->False veya False->True)
    flips = np.where(mask[1:] != mask[:-1])[0]

    if flips.size == 0:
        # bu gridde hiç geçiş yoksa NaN dönelim
        return np.nan

    # geçiş noktalarından |λ|'si en küçük olanı seç (λ=0 civarındaki threshold)
    cand_idx = flips
    best = cand_idx[np.argmin(np.abs(lambda_vals[cand_idx]))]

    # daha düzgün olsun diye iki noktanın ortasını al (istersen linear interp da yapabiliriz)
    lam_crit = 0.5 * (lambda_vals[best] + lambda_vals[best+1])
    return lam_crit

def plot_lambda_crit_vs_pairindex(
    N=51,
    beta=30.0,
    lambda_vals=None,
    periodic=True,
    eps=1e-4,
    savepath=None
):
    """
    Komşu pairler (i, i+1 mod N) için λ_crit değerlerini hesaplayıp
    pair index'e karşı plotlar.

    x-ekseni: i  (pair (i, i+1))
    y-ekseni: λ_crit  (concurrence ~ 0 olduğu sınır)
    """
    
    if savepath is None:
        savepath = FIG_DIR / f"lambda_critical_vs_pair_index{N}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)


    if lambda_vals is None:
        lambda_vals = np.linspace(-1.0, 1.0, 401)

    indices = np.arange(N)
    lambdas_crit = []

    for i in range(N):
        pair = (i, (i+1) % N)
        lam_c = find_lambda_crit_for_pair(pair, N, beta, lambda_vals,
                                          periodic=periodic, eps=eps)
        lambdas_crit.append(lam_c)

    lambdas_crit = np.array(lambdas_crit)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(indices, lambdas_crit, 'o-')
    ax.set_xlabel(r'pair index $i$ in $(i,i+1)$')
    ax.set_ylabel(r'$\lambda_\mathrm{crit}$ where $C\approx 0$')

    bc = "Periodic" if periodic else "Open"
    ax.set_title(fr'Critical $\lambda$ vs pair index (N={N}, $\beta={beta}$, {bc})')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

    return indices, lambdas_crit


#zero mode-bulk contribution plots and concuruence recreation

def diagonal_data_from_hamiltonian(N, beta, lam, periodic=True):
    H = build_ssh_hamiltonian(N, lam, periodic=periodic)
    eigvals, eigvecs = eigh(H)

    if beta > 100:
        f = expit(-beta * eigvals)
    else:
        x = np.clip(beta * eigvals, -700, 700)
        f = 1 / (np.exp(x) + 1)

    C = eigvecs @ np.diag(f) @ eigvecs.T
    return eigvals, eigvecs, f, C

def rho_from_abg(a, b, g):
    """
    Constructs the 4x4 two-site Gaussian RDM from
    C_A = [[a, g], [g, b]].
    Basis: |00>, |01>, |10>, |11>
    """
    rho00 = (1 - a) * (1 - b) - g**2
    rho01 = b * (1 - a) + g**2
    rho10 = a * (1 - b) + g**2
    rho11 = a * b - g**2

    rho = np.array([
        [rho00, 0,     0,     0],
        [0,     rho01, g,     0],
        [0,     g,     rho10, 0],
        [0,     0,     0,     rho11]
    ], dtype=float)

    # numerical cleanup
    rho[np.abs(rho) < 1e-14] = 0.0
    rho = 0.5 * (rho + rho.T)
    rho = rho / np.trace(rho)

    return rho



##semi analitical concurrence form bulk concurrence
def bulk_plus_trial_concurrence_pair(
    N=51,
    beta=30.0,
    lam=0.5,
    pair=(0, 1),
    use_actual_f0=False,
):
    """
    Interpretive bulk + trial-zero-mode approximation.

    C_A ≈ C_A^bulk + f0 * phi_A phi_A^T

    This is NOT the same as the semi-analytic reconstruction
    C_num - C_zm_num + C_zm_trial.
    It is a simpler analytic model to test how far the
    bulk + localized zero-mode picture goes.
    """

    i, j = pair

    # trial zero-mode profile
    phi = trial_near_zero_profile(N=N, lam=lam, normalize=True)

    # bulk correlation amplitude: parity determines r=0 or r=1
    # even i -> C1-like bond, odd i -> C2-like bond
    r = 0 if i % 2 == 0 else 1
    eta = eta_continuum(lam, beta=beta, r=r)

    # near-zero occupation
    if use_actual_f0:
        H = build_ssh_hamiltonian(N, lam, periodic=True)
        eigvals, _ = eigh(H)
        E0 = eigvals[int(np.argmin(np.abs(eigvals)))]
        f0 = 1 / (np.exp(beta * E0) + 1)
    else:
        f0 = 0.5

    a = 0.5 + f0 * phi[i]**2
    b = 0.5 + f0 * phi[j]**2
    g = -0.5 * eta + f0 * phi[i] * phi[j]

    rho = rho_from_abg(a, b, g)
    return concurrence_general(rho)

def plot_bulk_plus_trialzm_concurrence_vs_lambda(
    N=51,
    beta=30.0,
    lambda_vals=None,
    selected_pairs=None,
    periodic=True,
    use_actual_f0=False,
    savepath=None
):
    if lambda_vals is None:
        lambda_vals = np.linspace(-0.95, 0.95, 301)

    if selected_pairs is None:
        selected_pairs = [
            (0, 1),
            (N-1, 0),
            (1, 2),
            (N-2, N-1),
            (N//4, N//4 + 1),
            (N - N//4 - 1, N - N//4),
        ]

    if savepath is None:
        savepath = FIG_DIR / f"bulk_plus_trial_concurrence_N{N}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    num = len(selected_pairs)
    ncols = 2
    nrows = int(np.ceil(num / ncols))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 3.2*nrows),
        sharex=True,
        sharey=True
    )
    axs = np.array(axs).reshape(-1)

    for ax, pair in zip(axs, selected_pairs):

        vals = []

        for lam in lambda_vals:
            vals.append(
                bulk_plus_trial_concurrence_pair(
                    N=N,
                    beta=beta,
                    lam=lam,
                    pair=pair,
                    use_actual_f0=use_actual_f0
                )
            )

        ax.plot(lambda_vals, vals, lw=1.8, color="darkorange")

        ax.set_title(f"pair {pair}")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Concurrence")
        ax.grid(True)

    for ax in axs[num:]:
        ax.axis("off")

    bc_type = "Periodic" if periodic else "Open"
    fig.suptitle(
        rf"Bulk + trial zero-mode concurrence "
        rf"($N={N}$, $\beta={beta}$, {bc_type})",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

def plot_numeric_vs_bulk_plus_trial_concurrence(
    N=51,
    beta=30.0,
    lambda_vals=None,
    selected_pairs=None,
    periodic=True,
    use_actual_f0=False,
    savepath=None
):
    if lambda_vals is None:
        lambda_vals = np.linspace(-0.95, 0.95, 301)

    if selected_pairs is None:
        selected_pairs = [
            (0, 1),
            (N-1, 0),
            (1, 2),
            (N-2, N-1),
            (N//4, N//4 + 1),
            (N - N//4 - 1, N - N//4),
        ]

    if savepath is None:
        savepath = FIG_DIR / f"numeric_vs_bulk_plus_trial_N{N}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    num = len(selected_pairs)
    ncols = 2
    nrows = int(np.ceil(num / ncols))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 3.2*nrows),
        sharex=True,
        sharey=True
    )
    axs = np.array(axs).reshape(-1)

    for ax, pair in zip(axs, selected_pairs):
        numeric_vals = []
        bulk_trial_vals = []

        for lam in lambda_vals:
            rho_num = compute_rho(
                beta=beta,
                lam=lam,
                pair_idx=pair,
                N=N,
                periodic=periodic
            )
            numeric_vals.append(concurrence_general(rho_num))

            bulk_trial_vals.append(
                bulk_plus_trial_concurrence_pair(
                    N=N,
                    beta=beta,
                    lam=lam,
                    pair=pair,
                    use_actual_f0=use_actual_f0
                )
            )

        ax.plot(lambda_vals, numeric_vals, lw=1.8, label="numeric defect")
        ax.plot(lambda_vals, bulk_trial_vals, "--", lw=1.8, label="bulk + trial zm")

        ax.set_title(f"pair {pair}")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Concurrence")
        ax.grid(True)

    for ax in axs[num:]:
        ax.axis("off")

    fig.suptitle(
        rf"Numerical defect vs bulk + trial zero-mode concurrence "
        rf"($N={N}$, $\beta={beta}$)",
        fontsize=14
    )

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()








#####ENTANGLEMENT PHASE DIAGRAM NASIL DEĞİŞİYOR?





 #eta continium function

KS = np.linspace(0, 2*np.pi, 8192, endpoint=False) # k-grid for continuum integrals (periodic grid; endpoint=False to exclude 2pi)

def ssh_couplings(lam: float) -> tuple[float, float]:
    """t1 = 1 - lam,  t2 = 1 + lam"""
    return 1.0 - lam, 1.0 + lam

def h_complex_from_lam(lam: float, k: np.ndarray) -> np.ndarray:
    """
    Bloch off-diagonal term: h(k) = t1 + t2 e^{-ik}.
    Uses our convention t1=1-lam, t2=1+lam (via ssh_couplings).
    """
    t1, t2 = ssh_couplings(lam)
    return t1 + t2 * np.exp(-1j * k)

def abs_h_from_lam(lam: float, k: np.ndarray) -> np.ndarray:
    return np.abs(h_complex_from_lam(lam, k))

def phi_k_from_lam(lam: float, k: np.ndarray) -> np.ndarray:
    # phase in (-0, 2pi]
    return np.angle(h_complex_from_lam(lam, k))

def eta_continuum(lam: float, beta=np.inf, r: int = 0, ks: np.ndarray = KS) -> float:
    """
    Continuum correlator amplitude:
      η_r(λ,β) = (1/2π) ∫ dk  tanh(β|h(k)|/2) cos(rk + φ(k))
        where r=m-n is the site separation, and φ(k) = arg(h(k)).
    beta = np.inf -> tanh(...) -> 1 (ground-state limit)
    """
    ph = phi_k_from_lam(lam, ks)
    if np.isinf(beta):
        w = 1.0
        return np.cos(r*ks + ph).mean()
    else:
        w = np.tanh(0.5 * beta * abs_h_from_lam(lam, ks))
        return (w * np.cos(r*ks + ph)).mean()

def concurrence_from_eta(eta: float) -> float:
    """
    For the X-state structure used in the report:
      C = max(0, 1/2 (eta^2 + 2|eta| - 1))
    This matches the Kim&Cho active branch at T=0 (eta>=0),
    and stays safe if eta changes sign.
    """
    return max(0.0, 0.5 * (eta**2 + 2.0*np.abs(eta) - 1.0))


def bulk_concurrence_for_pair_parity(lam, beta, pair_index):
    """
    Uniform periodic SSH bulk concurrence reference.
    even pair index i -> C1-like bond, r=0
    odd  pair index i -> C2-like bond, r=1
    """
    r = 0 if pair_index % 2 == 0 else 1
    eta = eta_continuum(lam, beta=beta, r=r)
    return concurrence_from_eta(eta)

def plot_defect_delta_concurrence_heatmap(
    N=51,
    beta=30.0,
    lambda_vals=None,
    periodic=True,
    absolute=True,
    savepath=None,
    cmap=None,
):
    """
    Heatmap of Delta C_i(lambda) = C_defect(i,i+1) - C_bulk(i,i+1).

    If absolute=True, plots |Delta C|.
    If absolute=False, plots signed Delta C with a diverging colormap.
    """

    if lambda_vals is None:
        lambda_vals = np.linspace(-1.0, 1.0, 301)

    if savepath is None:
        tag = "abs" if absolute else "signed"
        savepath = FIG_DIR / f"delta_concurrence_heatmap_{tag}_N{N}_beta{beta}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    Delta = np.zeros((N, len(lambda_vals)))

    for p, lam in enumerate(lambda_vals):
        for i in range(N):
            pair = (i, (i + 1) % N)

            rho = compute_rho(
                beta=beta,
                lam=lam,
                pair_idx=pair,
                N=N,
                periodic=periodic
            )
            C_defect = concurrence_general(rho)

            C_bulk = bulk_concurrence_for_pair_parity(
                lam=lam,
                beta=beta,
                pair_index=i
            )

            Delta[i, p] = C_defect - C_bulk

    data = np.abs(Delta) if absolute else Delta

    if cmap is None:
        cmap = "magma" if absolute else "coolwarm"

    fig, ax = plt.subplots(figsize=(8, 5))

    if absolute:
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[lambda_vals[0], lambda_vals[-1], 0, N-1],
            cmap=cmap,
            vmin=0
        )
        cbar_label = r"$|\Delta C|$"
    else:
        vmax = np.nanmax(np.abs(data))
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[lambda_vals[0], lambda_vals[-1], 0, N-1],
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax
        )
        cbar_label = r"$\Delta C$"

    ax.axvline(0, color="white", ls="--", lw=1, alpha=0.8)

    # defect closure bond
    ax.axhline(N-1, color="white", ls=":", lw=1, alpha=0.8)
    ax.axhline(0, color="white", ls=":", lw=1, alpha=0.6)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"pair index $i$ in $(i,i+1)$")
    ax.set_title(
        rf"Defect-induced concurrence deviation "
        rf"($N={N}$, $\beta={beta}$, PBC)"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

    return lambda_vals, np.arange(N), Delta

##########Entanglement phase diagram



def find_lambda_crit_for_pair_beta(
    pair,
    N,
    beta,
    lambda_vals,
    periodic=True,
    eps=1e-4,
    boundary="near_zero"
):
    """
    Finds critical lambda values where concurrence crosses eps.

    boundary:
        "left"      -> left boundary of entangled region
        "right"     -> right boundary of entangled region
        "near_zero" -> crossing closest to lambda=0
        "both"      -> return (left, right)
    """

    conc_vals = []

    for lam in lambda_vals:
        rho = compute_rho(beta, lam, pair, N=N, periodic=periodic)
        conc_vals.append(concurrence_general(rho))

    conc_vals = np.array(conc_vals)
    mask = conc_vals > eps

    if not np.any(mask):
        return (np.nan, np.nan) if boundary == "both" else np.nan

    idx = np.where(mask)[0]
    i_left = idx[0]
    i_right = idx[-1]

    def interpolate_crossing(i0, i1):
        x0, x1 = lambda_vals[i0], lambda_vals[i1]
        y0, y1 = conc_vals[i0], conc_vals[i1]
        if np.isclose(y0, y1):
            return 0.5 * (x0 + x1)
        return x0 + (eps - y0) * (x1 - x0) / (y1 - y0)

    # left boundary
    if i_left == 0:
        lam_left = lambda_vals[0]
    else:
        lam_left = interpolate_crossing(i_left - 1, i_left)

    # right boundary
    if i_right == len(lambda_vals) - 1:
        lam_right = lambda_vals[-1]
    else:
        lam_right = interpolate_crossing(i_right, i_right + 1)

    if boundary == "left":
        return lam_left
    elif boundary == "right":
        return lam_right
    elif boundary == "both":
        return lam_left, lam_right
    elif boundary == "near_zero":
        candidates = [lam_left, lam_right]
        return candidates[int(np.argmin(np.abs(candidates)))]
    else:
        raise ValueError("boundary must be 'left', 'right', 'near_zero', or 'both'")


def plot_lambda_crit_vs_beta_pairs(
    selected_pairs,
    N=51,
    beta_vals=None,
    lambda_vals=None,
    periodic=True,
    eps=1e-4,
    boundary="near_zero"
):
    if beta_vals is None:
        beta_vals = np.geomspace(0.5, 30.0, 50)

    if lambda_vals is None:
        lambda_vals = np.linspace(-1.0, 1.0, 501)

    plt.figure(figsize=(7, 4.5))

    for pair in selected_pairs:
        lamc_vals = []

        for beta in beta_vals:
            lam_c = find_lambda_crit_for_pair_beta(
                pair=pair,
                N=N,
                beta=beta,
                lambda_vals=lambda_vals,
                periodic=periodic,
                eps=eps,
                boundary=boundary
            )
            lamc_vals.append(lam_c)

        plt.plot(beta_vals, lamc_vals, marker="o", ms=3, label=f"pair {pair}")

    plt.xscale("log")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\lambda_c$")
    plt.title(rf"Pair-dependent critical $\lambda_c(\beta)$ ($N={N}$, PBC)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()



## Add average bell plot as well then put into same file


################################################################
## Some control plots

def bulk_concurrence_pair(
    lam,
    beta=30.0,
    pair=(0, 1)
):
    """
    Uniform periodic bulk concurrence for the bond type corresponding to pair index.
    even i -> r=0, odd i -> r=1
    """
    i, j = pair
    r = 0 if i % 2 == 0 else 1

    eta = eta_continuum(lam, beta=beta, r=r)
    return concurrence_from_eta(eta)

def zero_mode_pair_correction_strength(
    N=51,
    lam=0.5,
    pair=(0, 1),
    f0=0.5,
    absolute=True
):
    """
    Zero-mode contribution to the off-diagonal pair correlation:
        f0 * phi_i * phi_j

    If absolute=True, returns |f0 phi_i phi_j|.
    """
    i, j = pair

    phi = trial_near_zero_profile(
        N=N,
        lam=lam,
        normalize=True
    )

    val = f0 * phi[i] * phi[j]

    return abs(val) if absolute else val



############ A NEW CLEAR PAGE TO ZERO MOD ECONTRIBUTION TO CONCURRENCE PLOTS

def concurrence_active_value(a, b, g):
    X = (1 - a) * (1 - b) - g**2
    Y = a * b - g**2
    if X <= 0 or Y <= 0:
        return 0.0
    Q = abs(g) - np.sqrt(X * Y)
    return 2 * max(0.0, Q)

def concurrence_gradients_abg(a, b, g, eps=1e-12):
    X = (1 - a) * (1 - b) - g**2
    Y = a * b - g**2

    Q = abs(g) - np.sqrt(max(X * Y, 0.0))

    # outside active concurrence region, derivative of max branch is set to zero
    if Q <= 0 or X <= eps or Y <= eps:
        return 0.0, 0.0, 0.0

    denom = np.sqrt(X * Y)

    Fa = ((1 - b) * Y - b * X) / denom
    Fb = ((1 - a) * Y - a * X) / denom

    if abs(g) < eps:
        sgn = 0.0
    else:
        sgn = np.sign(g)

    Fg = 2 * sgn + (2 * g * (X + Y)) / denom

    return Fa, Fb, Fg






    #### pek iyi görünmüyor bunun yerine başka ir analitik inceleme yapalım

def bulk_correlation_for_pair(lam, beta=30.0, pair=(0, 1)):
    """
    Uniform bulk correlation C_ij for nearest-neighbor-like pairs.
    Uses parity convention:
        even i -> r=0
        odd i  -> r=1
    """
    i, j = pair
    r = 0 if i % 2 == 0 else 1
    eta = eta_continuum(lam, beta=beta, r=r)
    return -0.5 * eta

def correlation_decomposition_pair(
    N=51,
    beta=30.0,
    lam=0.5,
    pair=(0, 1),
    periodic=True,
    use_trial=True,
):
    """
    Decomposes C_ij into:
        C_defect
        C_rest = C_defect - C_zero_num
        C_bulk
        C_zero_trial = f0 phi_i phi_j
        delta_C_rest = C_rest - C_bulk
    """

    eigvals, eigvecs, f, C_full = diagonal_data_from_hamiltonian(
        N=N,
        beta=beta,
        lam=lam,
        periodic=periodic
    )

    k0 = int(np.argmin(np.abs(eigvals)))
    psi0 = eigvecs[:, k0]
    f0 = f[k0]

    phi = trial_near_zero_profile(N=N, lam=lam, normalize=True)

    if np.dot(psi0, phi) < 0:
        phi = -phi

    i, j = pair

    C_defect = C_full[i, j]
    C_zero_num = f0 * psi0[i] * psi0[j]
    C_rest = C_defect - C_zero_num

    C_zero_trial = f0 * phi[i] * phi[j]
    C_bulk = bulk_correlation_for_pair(lam=lam, beta=beta, pair=pair)

    delta_C_rest = C_rest - C_bulk

    return {
        "C_defect": C_defect,
        "C_rest": C_rest,
        "C_bulk": C_bulk,
        "C_zero_num": C_zero_num,
        "C_zero_trial": C_zero_trial,
        "delta_C_rest": delta_C_rest,
        "f0": f0,
    }

def plot_correlation_decomposition_vs_lambda(
    N=51,
    beta=30.0,
    lambda_vals=None,
    selected_pairs=None,
    periodic=True,
    savepath=None,
):
    if lambda_vals is None:
        lambda_vals = np.linspace(-0.95, 0.95, 301)

    if selected_pairs is None:
        selected_pairs = [
            (0, 1),
            (0, 2),
            (1, 2),
            (12, 13),
            (24, 25),
        ]

    if savepath is None:
        savepath = FIG_DIR / f"correlation_decomposition_N{N}_beta{beta}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    num = len(selected_pairs)
    ncols = 2
    nrows = int(np.ceil(num / ncols))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4.5*ncols, 3.4*nrows),
        sharex=True
    )
    axs = np.array(axs).reshape(-1)

    for ax, pair in zip(axs, selected_pairs):

        C_defect_vals = []
        C_bulk_vals = []
        C_zero_vals = []
        delta_rest_vals = []
        recon_vals = []

        for lam in lambda_vals:
            out = correlation_decomposition_pair(
                N=N,
                beta=beta,
                lam=lam,
                pair=pair,
                periodic=periodic,
            )

            C_defect_vals.append(out["C_defect"])
            C_bulk_vals.append(out["C_bulk"])
            C_zero_vals.append(out["C_zero_trial"])
            delta_rest_vals.append(out["delta_C_rest"])

            recon_vals.append(
                out["C_bulk"] + out["C_zero_trial"] + out["delta_C_rest"]
            )

        ax.plot(lambda_vals, C_defect_vals, lw=2.0, label=r"$C^{defect}_{ij}$")
        ax.plot(lambda_vals, C_bulk_vals, "--", lw=1.6, label=r"$C^{bulk}_{ij}$")
        ax.plot(lambda_vals, C_zero_vals, ":", lw=2.0, label=r"$f_0\phi_i\phi_j$")
        ax.plot(lambda_vals, delta_rest_vals, "-.", lw=1.6, label=r"$\delta C^{rest}_{ij}$")
        ax.plot(lambda_vals, recon_vals, color="black", lw=0.9, alpha=0.6, label="reconstructed")

        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_title(f"pair {pair}")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$C_{ij}$ contribution")
        ax.grid(True, alpha=0.3)

    for ax in axs[num:]:
        ax.axis("off")

    fig.suptitle(
        rf"Correlation-matrix decomposition ($N={N}$, $\beta={beta}$)",
        fontsize=14
    )

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

#### concurrence-correlation relations, not linear thus step by step

def concurrence_decomposition_pair(
    N=51,
    beta=30.0,
    lam=0.5,
    pair=(0, 1),
    periodic=True,
):
    eigvals, eigvecs, f, C_full = diagonal_data_from_hamiltonian(
        N=N,
        beta=beta,
        lam=lam,
        periodic=periodic
    )

    k0 = int(np.argmin(np.abs(eigvals)))
    psi0 = eigvecs[:, k0]
    f0 = f[k0]

    phi = trial_near_zero_profile(N=N, lam=lam, normalize=True)
    if np.dot(psi0, phi) < 0:
        phi = -phi

    i, j = pair

    # --- FULL DEFECT ---
    C_sub_def = C_full[[i, j]][:, [i, j]]
    rho_def = rho_from_abg(C_sub_def[0,0], C_sub_def[1,1], C_sub_def[0,1])
    C_def = concurrence_general(rho_def)

    # --- BULK ---
    C_bulk_val = bulk_concurrence_for_pair_parity(
        lam=lam,
        beta=beta,
        pair_index=i
    )

    # --- BULK CORRELATION MATRIX ---
    a_bulk = 0.5
    b_bulk = 0.5
    g_bulk = bulk_correlation_for_pair(lam, beta, pair)

    rho_bulk = rho_from_abg(a_bulk, b_bulk, g_bulk)
    C_bulk = concurrence_general(rho_bulk)

    # --- BULK + ZERO MODE ---
    a_zm = a_bulk + f0 * phi[i]**2
    b_zm = b_bulk + f0 * phi[j]**2
    g_zm = g_bulk + f0 * phi[i] * phi[j]

    rho_zm = rho_from_abg(a_zm, b_zm, g_zm)
    C_bulk_zm = concurrence_general(rho_zm)

    return {
        "C_bulk": C_bulk,
        "C_bulk_zm": C_bulk_zm,
        "C_def": C_def,
        "delta_zm": C_bulk_zm - C_bulk,
        "delta_rest": C_def - C_bulk_zm
    }

def plot_concurrence_decomposition_vs_lambda(
    N=51,
    beta=30.0,
    lambda_vals=None,
    selected_pairs=None,
    periodic=True,
    savepath=None
):
    if lambda_vals is None:
        lambda_vals = np.linspace(-0.95, 0.95, 301)

    if selected_pairs is None:
        selected_pairs = [
            (0, 1),
            (1, 2),
            (12, 13),
            (24, 25),
        ]

    if savepath is None:
        savepath = FIG_DIR / f"concurrence_decomposition_N{N}_beta{beta}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    num = len(selected_pairs)
    ncols = 2
    nrows = int(np.ceil(num / ncols))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4.6*ncols, 3.5*nrows),
        sharex=True,
        sharey=True
    )
    axs = np.array(axs).reshape(-1)

    for ax, pair in zip(axs, selected_pairs):

        bulk_vals = []
        bulk_zm_vals = []
        defect_vals = []
        delta_zm_vals = []
        delta_rest_vals = []

        for lam in lambda_vals:
            out = concurrence_decomposition_pair(
                N=N,
                beta=beta,
                lam=lam,
                pair=pair,
                periodic=periodic
            )

            bulk_vals.append(out["C_bulk"])
            bulk_zm_vals.append(out["C_bulk_zm"])
            defect_vals.append(out["C_def"])
            delta_zm_vals.append(out["delta_zm"])
            delta_rest_vals.append(out["delta_rest"])

        ax.plot(lambda_vals, bulk_vals, lw=1.8, label=r"$C_W^{bulk}$")
        ax.plot(lambda_vals, bulk_zm_vals, "--", lw=1.8, label=r"$C_W^{bulk+zm}$")
        ax.plot(lambda_vals, defect_vals, lw=2.0, label=r"$C_W^{defect}$")

        ax.set_title(f"pair {pair}")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Concurrence")
        ax.grid(True, alpha=0.3)

    for ax in axs[num:]:
        ax.axis("off")

    fig.suptitle(
        rf"Concurrence decomposition: bulk, bulk+zero-mode, defect "
        rf"($N={N}$, $\beta={beta}$)",
        fontsize=14
    )

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

### deep dive into background 
def periodic_distance_from_defect(pair_index, N):
    """
    Distance of nearest-neighbor bond (i,i+1) from the defect closure region.
    Defect is taken near bonds i=0 and i=N-1.
    """
    return min(pair_index, N - 1 - pair_index)


def rest_background_deviation_for_bond(
    N=51,
    beta=30.0,
    lam=0.5,
    pair_index=0,
    periodic=True
):
    """
    Computes delta C_rest = C_rest - C_bulk for nearest-neighbor bond (i,i+1).
    """
    i = pair_index
    j = (i + 1) % N

    eigvals, eigvecs, f, C_full = diagonal_data_from_hamiltonian(
        N=N,
        beta=beta,
        lam=lam,
        periodic=periodic
    )

    k0 = int(np.argmin(np.abs(eigvals)))
    psi0 = eigvecs[:, k0]
    f0 = f[k0]

    C_zero_num = f0 * psi0[i] * psi0[j]
    C_rest = C_full[i, j] - C_zero_num

    C_bulk = bulk_correlation_for_pair(
        lam=lam,
        beta=beta,
        pair=(i, j)
    )

    return C_rest - C_bulk


def plot_rest_background_deviation_vs_distance(
    N=51,
    beta=30.0,
    lambda_list=(-0.75, -0.5, -0.25, 0.25, 0.5, 0.75),
    periodic=True,
    absolute=True,
    savepath=None
):
    if savepath is None:
        tag = "abs" if absolute else "signed"
        savepath = FIG_DIR / f"rest_background_deviation_vs_distance_{tag}_N{N}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    pair_indices = np.arange(N)
    distances = np.array([periodic_distance_from_defect(i, N) for i in pair_indices])

    for lam in lambda_list:
        vals = []

        for i in pair_indices:
            dC = rest_background_deviation_for_bond(
                N=N,
                beta=beta,
                lam=lam,
                pair_index=i,
                periodic=periodic
            )
            vals.append(abs(dC) if absolute else dC)

        vals = np.array(vals)

        # same distance appears twice; average them for cleaner curve
        d_unique = np.unique(distances)
        vals_avg = np.array([
            np.mean(vals[distances == d]) for d in d_unique
        ])

        ax.plot(
            d_unique,
            vals_avg,
            marker="o",
            ms=3,
            lw=1.6,
            label=fr"$\lambda={lam}$"
        )

    ax.set_xlabel("Distance from defect bond")
    ax.set_ylabel(r"$|\delta C^{\rm rest}_{i,i+1}|$" if absolute else r"$\delta C^{\rm rest}_{i,i+1}$")
    ax.set_title(
        rf"Background deformation vs distance "
        rf"($N={N}$, $\beta={beta}$)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_rest_background_deviation_logscale(
    N=51,
    beta=30.0,
    lambda_list=(-0.75, -0.5, -0.25, 0.25, 0.5, 0.75),
    periodic=True,
    eps=1e-12,
    savepath=None
):
    if savepath is None:
        savepath = FIG_DIR / f"rest_background_deviation_logscale_N{N}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    pair_indices = np.arange(N)
    distances = np.array([periodic_distance_from_defect(i, N) for i in pair_indices])

    for lam in lambda_list:
        vals = []

        for i in pair_indices:
            dC = rest_background_deviation_for_bond(
                N=N,
                beta=beta,
                lam=lam,
                pair_index=i,
                periodic=periodic
            )
            vals.append(abs(dC))

        vals = np.array(vals)

        d_unique = np.unique(distances)
        vals_avg = np.array([
            np.mean(vals[distances == d]) for d in d_unique
        ])

        ax.semilogy(
            d_unique,
            np.maximum(vals_avg, eps),
            marker="o",
            ms=3,
            lw=1.6,
            label=fr"$\lambda={lam}$"
        )

    ax.set_xlabel("Distance from defect bond")
    ax.set_ylabel(r"$|\delta C^{\rm rest}_{i,i+1}|$")
    ax.set_title(
        rf"Background deformation decay "
        rf"($N={N}$, $\beta={beta}$)"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

from scipy.optimize import curve_fit

def exp_decay(d, A, xi, c):
    return A * np.exp(-d / xi) + c

def fit_rest_background_decay(
    N=51,
    beta=30.0,
    lam=0.5,
    periodic=True,
    d_min=0,
    d_max=None,
    eps=1e-12,
    plot=True
):
    pair_indices = np.arange(N)
    distances = np.array([periodic_distance_from_defect(i, N) for i in pair_indices])

    vals = []
    for i in pair_indices:
        dC = rest_background_deviation_for_bond(
            N=N,
            beta=beta,
            lam=lam,
            pair_index=i,
            periodic=periodic
        )
        vals.append(abs(dC))

    vals = np.array(vals)

    d_unique = np.unique(distances)
    vals_avg = np.array([
        np.mean(vals[distances == d]) for d in d_unique
    ])

    if d_max is None:
        d_max = d_unique.max()

    mask = (d_unique >= d_min) & (d_unique <= d_max) & (vals_avg > eps)

    d_fit = d_unique[mask]
    y_fit = vals_avg[mask]

    popt, pcov = curve_fit(
        exp_decay,
        d_fit,
        y_fit,
        p0=(y_fit[0], 2.0, 0.0),
        maxfev=10000
    )

    A, xi, c = popt

    if plot:
        plt.figure(figsize=(6, 4))
        plt.semilogy(d_unique, np.maximum(vals_avg, eps), "o", label="data")
        plt.semilogy(d_unique, np.maximum(exp_decay(d_unique, *popt), eps), "-", label=fr"fit: $\xi={xi:.2f}$")
        plt.xlabel("Distance from defect bond")
        plt.ylabel(r"$|\delta C^{\rm rest}_{i,i+1}|$")
        plt.title(fr"Background deformation decay ($\lambda={lam}$)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    return {
        "lambda": lam,
        "A": A,
        "xi_bg": xi,
        "c": c,
        "d": d_unique,
        "values": vals_avg,
    }

def plot_zero_mode_profile_vs_distance(
    N=51,
    lambda_list=(-0.75, -0.5, -0.25, 0.25, 0.5, 0.75),
    center_site=0,
    savepath=None
):
    if savepath is None:
        savepath = FIG_DIR / f"zero_mode_profile_vs_distance_N{N}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    sites = np.arange(N)
    distances = np.minimum(sites, N - sites)

    for lam in lambda_list:
        phi = trial_near_zero_profile(N=N, lam=lam, normalize=True)
        prof = np.abs(phi)**2

        d_unique = np.unique(distances)
        prof_avg = np.array([
            np.mean(prof[distances == d]) for d in d_unique
        ])

        ax.plot(d_unique, prof_avg, marker="o", ms=3, lw=1.6, label=fr"$\lambda={lam}$")

    ax.set_xlabel("Distance from defect site")
    ax.set_ylabel(r"$|\phi_i|^2$")
    ax.set_title(fr"Trial near-zero mode profile vs distance ($N={N}$)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

def fit_zero_mode_profile_decay(
    N=51,
    lam=0.5,
    d_min=0,
    d_max=None,
    eps=1e-14,
    plot=True
):
    phi = trial_near_zero_profile(N=N, lam=lam, normalize=True)
    prof = np.abs(phi)**2

    sites = np.arange(N)
    distances = np.minimum(sites, N - sites)

    d_unique = np.unique(distances)
    prof_avg = np.array([
        np.mean(prof[distances == d]) for d in d_unique
    ])

    if d_max is None:
        d_max = d_unique.max()

    mask = (d_unique >= d_min) & (d_unique <= d_max) & (prof_avg > eps)

    d_fit = d_unique[mask]
    y_fit = prof_avg[mask]

    popt, pcov = curve_fit(
        exp_decay,
        d_fit,
        y_fit,
        p0=(y_fit[0], 2.0, 0.0),
        maxfev=10000
    )

    A, xi_prob, c = popt
    xi_wave = 2 * xi_prob

    if plot:
        plt.figure(figsize=(6, 4))
        plt.semilogy(d_unique, np.maximum(prof_avg, eps), "o", label="data")
        plt.semilogy(d_unique, np.maximum(exp_decay(d_unique, *popt), eps), "-", label=fr"fit: $\xi_{{prob}}={xi_prob:.2f}$")
        plt.xlabel("Distance from defect site")
        plt.ylabel(r"$|\phi_i|^2$")
        plt.title(fr"Zero-mode profile decay ($\lambda={lam}$)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    return {
        "lambda": lam,
        "A": A,
        "xi_prob": xi_prob,
        "xi_wave": xi_wave,
        "c": c,
        "d": d_unique,
        "values": prof_avg,
    }

def fit_exp_loglinear(d, y, d_min=0, d_max=8, eps=1e-14):
    d = np.asarray(d)
    y = np.asarray(y)

    mask = (d >= d_min) & (d <= d_max) & (y > eps)

    d_fit = d[mask]
    y_fit = y[mask]

    logy = np.log(y_fit)

    # log y = log A - d/xi
    slope, intercept = np.polyfit(d_fit, logy, 1)

    xi = -1 / slope
    A = np.exp(intercept)

    y_pred = A * np.exp(-d / xi)

    return A, xi, d_fit, y_fit, y_pred

def fit_rest_background_decay_loglinear(
    N=51,
    beta=30.0,
    lam=0.5,
    periodic=True,
    d_min=0,
    d_max=8,
    eps=1e-12
):
    pair_indices = np.arange(N)
    distances = np.array([periodic_distance_from_defect(i, N) for i in pair_indices])

    vals = []
    for i in pair_indices:
        dC = rest_background_deviation_for_bond(
            N=N,
            beta=beta,
            lam=lam,
            pair_index=i,
            periodic=periodic
        )
        vals.append(abs(dC))

    vals = np.array(vals)

    d_unique = np.unique(distances)
    vals_avg = np.array([
        np.mean(vals[distances == d]) for d in d_unique
    ])

    A, xi, d_fit, y_fit, y_pred = fit_exp_loglinear(
        d_unique, vals_avg, d_min=d_min, d_max=d_max, eps=eps
    )

    plt.figure(figsize=(6, 4))
    plt.semilogy(d_unique, np.maximum(vals_avg, eps), "o", label="data")
    plt.semilogy(d_unique, np.maximum(y_pred, eps), "-", label=fr"log-fit: $\xi={xi:.2f}$")
    plt.xlabel("Distance from defect bond")
    plt.ylabel(r"$|\delta C^{\rm rest}_{i,i+1}|$")
    plt.title(fr"Background deformation decay ($\lambda={lam}$)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return {
        "lambda": lam,
        "A": A,
        "xi_bg": xi,
        "d": d_unique,
        "values": vals_avg
    }

def theoretical_zero_mode_lengths(lam):
    """
    q < 1 olacak şekilde localization ratio.
    amplitude: |phi| ~ q^d = exp(-d/xi_amp)
    probability: |phi|^2 ~ q^(2d) = exp(-d/xi_prob)
    """
    t1 = 1 - lam
    t2 = 1 + lam

    if np.isclose(lam, 0):
        return np.nan, np.nan, np.nan

    r = abs(t1 / t2)
    q = r if r < 1 else 1 / r

    xi_amp = -1 / np.log(q)
    xi_prob = -1 / (2 * np.log(q))

    return q, xi_amp, xi_prob

def extract_xi_bg_vs_lambda(
    N=51,
    beta=30.0,
    lambda_list=(-0.75, -0.5, -0.25, 0.25, 0.5, 0.75),
    d_min=0,
    d_max=8,
    periodic=True,
    eps=1e-12,
):
    results = []

    for lam in lambda_list:
        out = fit_rest_background_decay_loglinear(
            N=N,
            beta=beta,
            lam=lam,
            periodic=periodic,
            d_min=d_min,
            d_max=d_max,
            eps=eps
        )

        q, xi_amp, xi_prob = theoretical_zero_mode_lengths(lam)

        results.append({
            "lambda": lam,
            "xi_bg": out["xi_bg"],
            "q": q,
            "xi_amp": xi_amp,
            "xi_prob": xi_prob,
        })

    return results

def plot_xi_bg_vs_zero_mode_lengths(
    N=51,
    beta=30.0,
    lambda_list=(-0.75, -0.5, -0.25, 0.25, 0.5, 0.75),
    d_min=0,
    d_max=8,
    periodic=True,
    savepath=None,
):
    results = extract_xi_bg_vs_lambda(
        N=N,
        beta=beta,
        lambda_list=lambda_list,
        d_min=d_min,
        d_max=d_max,
        periodic=periodic
    )

    lambdas = np.array([r["lambda"] for r in results])
    xi_bg = np.array([r["xi_bg"] for r in results])
    xi_amp = np.array([r["xi_amp"] for r in results])
    xi_prob = np.array([r["xi_prob"] for r in results])

    if savepath is None:
        savepath = FIG_DIR / f"xi_bg_vs_zero_mode_N{N}_beta{beta}.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.5, 4.2))

    plt.plot(lambdas, xi_bg, "o-", label=r"$\xi_{\rm bg}$ from $\delta C^{\rm rest}$")
    plt.plot(lambdas, xi_amp, "s--", label=r"$\xi_{\rm amp}= -1/\ln q$")
    plt.plot(lambdas, xi_prob, "d--", label=r"$\xi_{\rm prob}= -1/(2\ln q)$")

    plt.axvline(0, color="gray", ls=":", lw=1)

    plt.xlabel(r"$\lambda$")
    plt.ylabel("length scale")
    plt.title(rf"Background deformation length vs zero-mode length ($N={N}$, $\beta={beta}$)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.show()

    return results

def plot_xi_ratios(
    results,
    savepath=None
):
    lambdas = np.array([r["lambda"] for r in results])
    xi_bg = np.array([r["xi_bg"] for r in results])
    xi_amp = np.array([r["xi_amp"] for r in results])
    xi_prob = np.array([r["xi_prob"] for r in results])

    ratio_amp = xi_bg / xi_amp
    ratio_prob = xi_bg / xi_prob

    if savepath is None:
        savepath = FIG_DIR / "xi_bg_zero_mode_ratios.pdf"
    else:
        savepath = Path(savepath).resolve()

    savepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(lambdas, ratio_amp, "o-", label=r"$\xi_{\rm bg}/\xi_{\rm amp}$")
    ax.plot(lambdas, ratio_prob, "s--", label=r"$\xi_{\rm bg}/\xi_{\rm prob}$")

    ax.axhline(1, color="gray", ls=":", lw=1)
    ax.axvline(0, color="gray", ls=":", lw=1)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("ratio")
    ax.set_title("Background length relative to zero-mode length")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

    return {
        "lambda": lambdas,
        "xi_bg_over_xi_amp": ratio_amp,
        "xi_bg_over_xi_prob": ratio_prob
    }


#### Analitik yaklaşımwith contour integral

def fermi_complex(z, beta):
    return 1.0 / (np.exp(beta * z) + 1.0)

def correlation_matrix_contour(
    N=51,
    lam=0.5,
    beta=30.0,
    periodic=True,
    n_theta=2048,
    margin=0.2,
    imag_height=None,
    verbose=True
):
    """
    Computes C = (1/2πi) ∮ dz f(z) (zI-H)^(-1)
    using an elliptical contour around the real spectrum.
    """

    H = build_ssh_hamiltonian(N, lam, periodic=periodic)
    eigvals = np.linalg.eigvalsh(H)

    Emax = np.max(np.abs(eigvals))
    a = Emax + margin

    # Fermi function has poles at z = i*pi(2n+1)/beta.
    # So keep contour height below pi/beta.
    if imag_height is None:
        imag_height = 0.5 * np.pi / beta

    b = imag_height

    I = np.eye(N, dtype=complex)
    C = np.zeros((N, N), dtype=complex)

    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    dtheta = 2*np.pi / n_theta

    for th in thetas:
        z = a * np.cos(th) + 1j * b * np.sin(th)
        dz_dtheta = -a * np.sin(th) + 1j * b * np.cos(th)

        Gz = np.linalg.inv(z * I - H)

        C += fermi_complex(z, beta) * Gz * dz_dtheta * dtheta

    C *= 1.0 / (2j * np.pi)

    # numerical cleanup
    C = 0.5 * (C + C.conj().T)

    if verbose:
        print("Contour correlation matrix computed.")
        print("Emax =", Emax)
        print("ellipse real semi-axis a =", a)
        print("ellipse imag semi-axis b =", b)
        print("Fermi first pole distance pi/beta =", np.pi / beta)
        print("Hermiticity check:", np.allclose(C, C.conj().T, atol=1e-8))

    return C

def selected_pair_correlation_contour(
    beta=30.0,
    lam=0.5,
    pair_idx=(0, 1),
    N=51,
    periodic=True,
    n_theta=2048,
    verbose=True
):
    C = correlation_matrix_contour(
        N=N,
        lam=lam,
        beta=beta,
        periodic=periodic,
        n_theta=n_theta,
        verbose=False
    )

    i, j = pair_idx
    C_sub = C[[i, j]][:, [i, j]]

    if verbose:
        print(f"\nSelected pair: {pair_idx}")
        print("C_sub =")
        print(C_sub)

        print("\nElements:")
        print(f"C_{i}{i} =", C_sub[0, 0])
        print(f"C_{i}{j} =", C_sub[0, 1])
        print(f"C_{j}{i} =", C_sub[1, 0])
        print(f"C_{j}{j} =", C_sub[1, 1])

        print("\neta if applicable:")
        print(f"eta_{i}{j} = -2 C_{i}{j} =", -2 * C_sub[0, 1])

    return C_sub, C

def compare_contour_vs_spectral(
    N=51,
    lam=0.5,
    beta=30.0,
    periodic=True,
    pair_idx=(0, 1),
    n_theta=2048
):
    C_contour = correlation_matrix_contour(
        N=N,
        lam=lam,
        beta=beta,
        periodic=periodic,
        n_theta=n_theta,
        verbose=False
    )

    eigvals, eigvecs, f, C_spectral = diagonal_data_from_hamiltonian(
        N=N,
        beta=beta,
        lam=lam,
        periodic=periodic
    )

    diff = np.max(np.abs(C_contour - C_spectral))

    i, j = pair_idx
    CA_contour = C_contour[[i, j]][:, [i, j]]
    CA_spectral = C_spectral[[i, j]][:, [i, j]]

    print("Max |C_contour - C_spectral| =", diff)

    print("\nC_A from contour:")
    print(CA_contour)

    print("\nC_A from spectral:")
    print(CA_spectral)

    print("\nDifference:")
    print(CA_contour - CA_spectral)

    return C_contour, C_spectral

## contour aracılığıyla defect-bulk durumu 

def selected_pair_delta_correlation(
    N_def=51,
    N_bulk=200,
    beta=30.0,
    lam=0.5,
    pair_idx=(0, 1),
    n_theta=4096,
    verbose=True
):
    """
    Compares defect-chain pair correlation with a clean even periodic bulk reference.

    Defect:
        odd-N PBC chain

    Bulk reference:
        large even-N PBC chain, same local bond parity.
    """

    i, j = pair_idx

    # defect chain correlation matrix by contour
    
    C_def = correlation_matrix_contour(
        N=N_def,
        lam=lam,
        beta=beta,
        periodic=True,
        n_theta=n_theta,
        verbose=False
    )

    Cij_def = C_def[i, j]

    # bulk reference: use large even periodic chain
    C_bulk = correlation_matrix_contour(
        N=N_bulk,
        lam=lam,
        beta=beta,
        periodic=True,
        n_theta=n_theta,
        verbose=False
    )

    # Match local parity of pair
    # For nearest neighbors, bond i -> i+1:
    # even i: t1-like, odd i: t2-like
    ib = i % 2
    jb = (ib + (j - i)) % N_bulk

    Cij_bulk = C_bulk[ib, jb]

    delta = Cij_def - Cij_bulk

    if verbose:
        print(f"Defect chain N={N_def}, pair={pair_idx}")
        print(f"C_def[{i},{j}] =", Cij_def)

        print(f"\nBulk reference N={N_bulk}, matched pair=({ib},{jb})")
        print(f"C_bulk[{ib},{jb}] =", Cij_bulk)

        print("\nDelta C = C_def - C_bulk")
        print(delta)

        print("\neta values if applicable:")
        print("eta_def  =", -2 * Cij_def)
        print("eta_bulk =", -2 * Cij_bulk)
        print("Delta eta =", -2 * delta)

    return delta, Cij_def, Cij_bulk

def plot_delta_correlation_from_defect(
    N_def=51,
    N_bulk=200,
    beta=30.0,
    lam=0.5,
    ref_site=0,
    n_theta=4096,
    use_abs=True
):
    """
    Plots Delta C_{ref_site,j} = C_def(ref_site,j) - C_bulk(matched)
    as a function of distance/site index.
    """

    C_def = correlation_matrix_contour(
        N=N_def,
        lam=lam,
        beta=beta,
        periodic=True,
        n_theta=n_theta,
        verbose=False
    )

    C_bulk = correlation_matrix_contour(
        N=N_bulk,
        lam=lam,
        beta=beta,
        periodic=True,
        n_theta=n_theta,
        verbose=False
    )

    js = np.arange(N_def)
    deltas = np.zeros(N_def, dtype=complex)

    for idx, j in enumerate(js):
        # defect pair
        Cij_def = C_def[ref_site, j]

        # match displacement and parity in even bulk chain
        d = j - ref_site
        ib = ref_site % 2
        jb = (ib + d) % N_bulk

        Cij_bulk = C_bulk[ib, jb]

        deltas[idx] = Cij_def - Cij_bulk

    y = np.abs(deltas) if use_abs else deltas.real

    plt.figure(figsize=(7, 4))
    plt.plot(js, y, "o-", ms=4)

    plt.axvline(ref_site, color="gray", ls="--", lw=0.8, label="ref site")
    plt.axvline(N_def - 1, color="red", ls=":", lw=1.0, label="closing bond site")

    plt.xlabel(r"site $j$")
    ylabel = r"$|\Delta C_{0j}|$" if use_abs else r"$\mathrm{Re}\,\Delta C_{0j}$"
    plt.ylabel(ylabel)

    plt.title(
        rf"Defect-induced correction to correlations "
        rf"($N={N_def}$, $\lambda={lam}$, $\beta={beta}$)"
    )

    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return js, deltas

def plot_delta_correlation_centered(
    N_def=51,
    N_bulk=200,
    beta=30.0,
    lam=0.5,
    ref_site=0,
    n_theta=4096,
    use_abs=True
):
    C_def = correlation_matrix_contour(
        N=N_def,
        lam=lam,
        beta=beta,
        periodic=True,
        n_theta=n_theta,
        verbose=False
    )

    C_bulk = correlation_matrix_contour(
        N=N_bulk,
        lam=lam,
        beta=beta,
        periodic=True,
        n_theta=n_theta,
        verbose=False
    )

    js = np.arange(N_def)
    deltas = np.zeros(N_def, dtype=complex)

    for idx, j in enumerate(js):
        Cij_def = C_def[ref_site, j]

        d = j - ref_site
        ib = ref_site % 2
        jb = (ib + d) % N_bulk

        Cij_bulk = C_bulk[ib, jb]

        deltas[idx] = Cij_def - Cij_bulk

    # center site 0 in the middle
    shift = N_def // 2 - ref_site
    deltas_centered = np.roll(deltas, shift)

    x = np.arange(N_def) - N_def // 2

    y = np.abs(deltas_centered) if use_abs else deltas_centered.real

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, "o-", ms=4)

    plt.axvline(0, color="gray", ls="--", lw=0.8, label="site 0")
    plt.axvline(-1, color="red", ls=":", lw=1.0, label="site 50")
    plt.axvline(1, color="orange", ls=":", lw=1.0, label="site 1")

    plt.xlabel("distance from site 0")
    ylabel = r"$|\Delta C_{0j}|$" if use_abs else r"$\mathrm{Re}\,\Delta C_{0j}$"
    plt.ylabel(ylabel)

    plt.title(
        rf"Centered defect correction "
        rf"($N={N_def}$, $\lambda={lam}$, $\beta={beta}$)"
    )

    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return x, deltas_centered

#birden fazla lambda ile gösterimi 
def plot_centered_delta_correlation_multi_lambda(
    N_def=51,
    N_bulk=200,
    beta=30.0,
    lambda_list=(0.2, 0.5, 0.8),
    ref_site=0,
    n_theta=4096,
    use_abs=True
):
    plt.figure(figsize=(7, 4))

    for lam in lambda_list:
        C_def = correlation_matrix_contour(
            N=N_def,
            lam=lam,
            beta=beta,
            periodic=True,
            n_theta=n_theta,
            verbose=False
        )

        C_bulk = correlation_matrix_contour(
            N=N_bulk,
            lam=lam,
            beta=beta,
            periodic=True,
            n_theta=n_theta,
            verbose=False
        )

        js = np.arange(N_def)
        deltas = np.zeros(N_def, dtype=complex)

        for idx, j in enumerate(js):
            Cij_def = C_def[ref_site, j]

            d = j - ref_site
            ib = ref_site % 2
            jb = (ib + d) % N_bulk

            Cij_bulk = C_bulk[ib, jb]
            deltas[idx] = Cij_def - Cij_bulk

        shift = N_def // 2 - ref_site
        deltas_centered = np.roll(deltas, shift)
        x = np.arange(N_def) - N_def // 2

        y = np.abs(deltas_centered) if use_abs else deltas_centered.real

        plt.plot(x, y, "o-", ms=3.5, lw=1.4, label=fr"$\lambda={lam}$")

    plt.axvline(0, color="gray", ls="--", lw=0.8, label="site 0")
    plt.axvline(-1, color="red", ls=":", lw=1.0, label="site 50")
    plt.axvline(1, color="orange", ls=":", lw=1.0, label="site 1")

    plt.xlabel("distance from site 0")
    ylabel = r"$|\Delta C_{0j}|$" if use_abs else r"$\mathrm{Re}\,\Delta C_{0j}$"
    plt.ylabel(ylabel)

    plt.title(
        rf"Centered defect correction for different $\lambda$ "
        rf"($N={N_def}$, $\beta={beta}$)"
    )

    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

# contour correlationdan concurrence calculation

def rho_from_C_sub_peschel(C_sub, eps=1e-10, verbose=False):
    """
    Constructs the two-site reduced density matrix from a 2x2 correlation
    matrix using Peschel's formula.

    C_sub: 2x2 correlation matrix for selected pair.
    Basis: |00>, |01>, |10>, |11>
    """

    # Hermitian cleanup
    C_sub = 0.5 * (C_sub + C_sub.conj().T)

    # diagonalize C_A
    evals_C, U = eigh(C_sub)
    evals_C = np.clip(evals_C, eps, 1 - eps)

    if verbose:
        print("C_sub =")
        print(C_sub)
        print("\nEigenvalues of C_sub:")
        print(evals_C)

    # Peschel formula: h_A = log((1-C_A)/C_A)
    H_red = U @ np.diag(np.log((1 - evals_C) / evals_C)) @ U.conj().T

    h11 = H_red[0, 0]
    h12 = H_red[0, 1]
    h21 = H_red[1, 0]
    h22 = H_red[1, 1]

    # two-mode occupation basis: |00>, |01>, |10>, |11>
    H_occ = np.array([
        [0,   0,   0,   0],
        [0, h22, h21,   0],
        [0, h12, h11,   0],
        [0,   0,   0, h11 + h22]
    ], dtype=complex)

    evals_occ, evecs_occ = eigh(H_occ)

    exp_neg_evals = np.exp(-evals_occ)
    Z = np.sum(exp_neg_evals)

    rho = evecs_occ @ np.diag(exp_neg_evals / Z) @ evecs_occ.conj().T

    # cleanup
    rho[np.abs(rho) < 1e-14] = 0.0
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)

    if verbose:
        print("\nEntanglement Hamiltonian h_A:")
        print(H_red)
        print("\nOccupation basis H_occ:")
        print(H_occ)
        print("\nReduced density matrix rho:")
        print(rho)

    return rho

def concurrence_from_contour_peschel(
    N=51,
    lam=0.5,
    beta=30.0,
    pair_idx=(0, 1),
    periodic=True,
    n_theta=4096,
    verbose=True
):
    C = correlation_matrix_contour(
        N=N,
        lam=lam,
        beta=beta,
        periodic=periodic,
        n_theta=n_theta,
        verbose=False
    )

    i, j = pair_idx
    C_sub = C[[i, j]][:, [i, j]]

    rho = rho_from_C_sub_peschel(
        C_sub,
        verbose=verbose
    )

    conc = concurrence_general(rho)

    if verbose:
        print(f"\nPair: {pair_idx}")
        print("\nConcurrence:")
        print(conc)

    return conc, rho, C_sub

def concurrence_vs_lambda_contour(
    lambda_vals=np.linspace(-1, 1, 101),
    beta=30.0,
    pair_idx=(0, 1),
    N=51,
    periodic=True,
    n_theta=2048,
    verbose=False
):
    concs = []

    for lam in lambda_vals:

        try:
            conc, rho, C_sub = concurrence_from_contour_peschel(
                N=N,
                lam=lam,
                beta=beta,
                pair_idx=pair_idx,
                periodic=periodic,
                n_theta=n_theta,
                verbose=False
            )

            concs.append(np.real_if_close(conc))

            if verbose:
                print(f"lambda={lam:.3f}  concurrence={conc:.8f}")

        except Exception as e:
            print(f"lambda={lam:.3f} failed: {e}")
            concs.append(np.nan)

    concs = np.array(concs, dtype=float)

    plt.figure(figsize=(7,4))

    plt.plot(
        lambda_vals,
        concs,
        lw=2
    )

    plt.axvline(0, color='gray', ls='--', lw=1)

    plt.xlabel(r'$\lambda$')
    plt.ylabel('Concurrence')

    i, j = pair_idx

    plt.title(
        rf'Concurrence vs $\lambda$ '
        rf'for pair ({i},{j}) '
        rf'($N={N}$, $\beta={beta}$)'
    )

    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return lambda_vals, concs
#### cnucrrence umericle contourdan gelenlerin ayrılması 

def plot_numeric_vs_contour_concurrence_validation(
    N=51,
    beta=30.0,
    lambda_vals=None,
    periodic=True,
    selected_pairs=None,
    n_theta=2048
):
    if lambda_vals is None:
        lambda_vals = np.linspace(-1, 1, 121)

    if selected_pairs is None:
        selected_pairs = [(0,1), (50,0), (1,2), (49,50)]

    num = len(selected_pairs)
    ncols = 2
    nrows = int(np.ceil(num / ncols))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 3.2*nrows),
        sharex=True,
        sharey=True
    )

    axs = np.array(axs).reshape(-1)

    for ax, pair in zip(axs, selected_pairs):

        conc_numeric = []
        conc_contour = []

        for lam in lambda_vals:
            # old spectral/Peschel route
            rho_num = compute_rho(
                beta=beta,
                lam=lam,
                pair_idx=pair,
                N=N,
                periodic=periodic
            )
            conc_numeric.append(concurrence_general(rho_num))

            # contour/Peschel route
            conc_c, _, _ = concurrence_from_contour_peschel(
                N=N,
                lam=lam,
                beta=beta,
                pair_idx=pair,
                periodic=periodic,
                n_theta=n_theta,
                verbose=False
            )
            conc_contour.append(conc_c)

        conc_numeric = np.array(conc_numeric)
        conc_contour = np.array(conc_contour)

        ax.plot(lambda_vals, conc_numeric, lw=2, label="spectral")
        ax.plot(lambda_vals, conc_contour, "--", lw=1.8, label="contour")

        maxdiff = np.nanmax(np.abs(conc_numeric - conc_contour))

        ax.set_title(f"pair {pair}\nmax diff={maxdiff:.1e}")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Concurrence")
        ax.grid(True)

    for ax in axs[num:]:
        ax.axis("off")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    fig.suptitle(
        rf"Validation: spectral vs contour concurrence "
        rf"($N={N}$, $\beta={beta}$)",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    plt.show()

### zero mode ve contour defectin farkına bakalım zero modedan gelen katkıyı correlation matrix üzerinde

def zero_mode_correlation_contribution(
    N=51,
    lam=0.5,
    beta=30.0,
    periodic=True,
    use_actual_f0=True
):
    H = build_ssh_hamiltonian(N, lam, periodic=periodic)
    E, U = eigh(H)

    k0 = int(np.argmin(np.abs(E)))
    E0 = E[k0]
    psi0 = U[:, k0]

    if use_actual_f0:
        f0 = 1 / (np.exp(beta * E0) + 1)
    else:
        f0 = 0.5

    C0 = f0 * np.outer(psi0, psi0)

    return C0, E0, psi0, f0

def decompose_defect_correlation(
    N_def=51,
    N_bulk=200,
    lam=0.5,
    beta=30.0,
    n_theta=4096,
    periodic=True
):
    C_def = correlation_matrix_contour(
        N=N_def,
        lam=lam,
        beta=beta,
        periodic=periodic,
        n_theta=n_theta,
        verbose=False
    )

    C_bulk_big = correlation_matrix_contour(
        N=N_bulk,
        lam=lam,
        beta=beta,
        periodic=True,
        n_theta=n_theta,
        verbose=False
    )

    C_bulk_matched = np.zeros_like(C_def, dtype=complex)

    for i in range(N_def):
        for j in range(N_def):
            d = j - i
            ib = i % 2
            jb = (ib + d) % N_bulk
            C_bulk_matched[i, j] = C_bulk_big[ib, jb]

    DeltaC = C_def - C_bulk_matched

    C0, E0, psi0, f0 = zero_mode_correlation_contribution(
        N=N_def,
        lam=lam,
        beta=beta,
        periodic=periodic
    )

    C_bg_def = DeltaC - C0

    return {
        "C_def": C_def,
        "C_bulk": C_bulk_matched,
        "DeltaC": DeltaC,
        "C0": C0,
        "C_bg_def": C_bg_def,
        "E0": E0,
        "psi0": psi0,
        "f0": f0
    }

def plot_C0j_decomposition(
    N_def=51,
    N_bulk=200,
    lam=0.5,
    beta=30.0,
    ref_site=0,
    n_theta=4096,
    use_abs=False
):
    data = decompose_defect_correlation(
        N_def=N_def,
        N_bulk=N_bulk,
        lam=lam,
        beta=beta,
        n_theta=n_theta
    )

    DeltaC = data["DeltaC"]
    C0 = data["C0"]
    C_bg_def = data["C_bg_def"]

    js = np.arange(N_def)

    shift = N_def // 2 - ref_site
    x = np.arange(N_def) - N_def // 2

    y_delta = np.roll(DeltaC[ref_site, :], shift)
    y_zero = np.roll(C0[ref_site, :], shift)
    y_bg = np.roll(C_bg_def[ref_site, :], shift)

    if use_abs:
        y_delta = np.abs(y_delta)
        y_zero = np.abs(y_zero)
        y_bg = np.abs(y_bg)
        ylabel = r"absolute value"
    else:
        y_delta = y_delta.real
        y_zero = y_zero.real
        y_bg = y_bg.real
        ylabel = r"real part"

    plt.figure(figsize=(7,4))

    plt.plot(x, y_delta, "o-", ms=4, label=r"$\Delta C_{0j}$")
    plt.plot(x, y_zero, "s--", ms=4, label=r"$C^0_{0j}$")
    plt.plot(x, y_bg, "d-.", ms=4, label=r"$\Delta C_{0j}-C^0_{0j}$")

    plt.axvline(0, color="gray", ls="--", lw=0.8, label="site 0")
    plt.axvline(-1, color="red", ls=":", lw=1.0, label="site 50")
    plt.axvline(1, color="orange", ls=":", lw=1.0, label="site 1")

    plt.xlabel("distance from site 0")
    plt.ylabel(ylabel)

    plt.title(
        rf"Defect correction decomposition "
        rf"($N={N_def}$, $\lambda={lam}$, $\beta={beta}$)"
    )

    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    print("E0 =", data["E0"])
    print("f0 =", data["f0"])

    return data

def plot_C0j_decomposition_ratio(
    N_def=51,
    N_bulk=200,
    lam=0.5,
    beta=30.0,
    ref_site=0,
    n_theta=4096,
    eps=1e-12
):
    data = decompose_defect_correlation(
        N_def=N_def,
        N_bulk=N_bulk,
        lam=lam,
        beta=beta,
        n_theta=n_theta
    )

    DeltaC = data["DeltaC"]
    C0 = data["C0"]
    C_bg = data["C_bg_def"]

    shift = N_def // 2 - ref_site
    x = np.arange(N_def) - N_def // 2

    delta = np.roll(DeltaC[ref_site, :], shift)
    zero = np.roll(C0[ref_site, :], shift)
    bg = np.roll(C_bg[ref_site, :], shift)

    ratio = np.abs(zero) / (np.abs(delta) + eps)

    fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    axs[0].plot(x, np.abs(delta), "o-", ms=4, label=r"$|\Delta C_{0j}|$")
    axs[0].plot(x, np.abs(zero), "s--", ms=4, label=r"$|C^0_{0j}|$")
    axs[0].plot(x, np.abs(bg), "d-.", ms=4, label=r"$|\Delta C_{0j}-C^0_{0j}|$")

    axs[0].set_ylabel("absolute value")
    axs[0].set_title(
        rf"Zero-mode decomposition "
        rf"($N={N_def}$, $\lambda={lam}$, $\beta={beta}$)"
    )
    axs[0].grid(True, ls="--", alpha=0.4)
    axs[0].legend(frameon=False)

    axs[1].plot(x, ratio, "o-", ms=4)
    axs[1].axhline(1, color="gray", ls="--", lw=1)

    axs[1].set_xlabel("distance from site 0")
    axs[1].set_ylabel(r"$|C^0_{0j}|/|\Delta C_{0j}|$")
    axs[1].grid(True, ls="--", alpha=0.4)

    for ax in axs:
        ax.axvline(0, color="gray", ls="--", lw=0.8)
        ax.axvline(-1, color="red", ls=":", lw=1.0)
        ax.axvline(1, color="orange", ls=":", lw=1.0)

    plt.tight_layout()
    plt.show()

    print("E0 =", data["E0"])
    print("f0 =", data["f0"])

    return data, x, ratio

def plot_C0j_signed_cancellation_zoom(
    N_def=51,
    N_bulk=200,
    lam=0.5,
    beta=30.0,
    ref_site=0,
    n_theta=4096,
    zoom_radius=8
):
    data = decompose_defect_correlation(
        N_def=N_def,
        N_bulk=N_bulk,
        lam=lam,
        beta=beta,
        n_theta=n_theta
    )

    DeltaC = data["DeltaC"]
    C0 = data["C0"]
    C_bg = data["C_bg_def"]

    shift = N_def // 2 - ref_site
    x = np.arange(N_def) - N_def // 2

    delta = np.roll(DeltaC[ref_site, :], shift).real
    zero = np.roll(C0[ref_site, :], shift).real
    bg = np.roll(C_bg[ref_site, :], shift).real

    mask = np.abs(x) <= zoom_radius

    plt.figure(figsize=(7, 4.5))

    plt.plot(x[mask], delta[mask], "o-", ms=5, lw=1.8, label=r"$\Delta C_{0j}$")
    plt.plot(x[mask], zero[mask], "s--", ms=5, lw=1.8, label=r"$C^0_{0j}$")
    plt.plot(x[mask], bg[mask], "d-.", ms=5, lw=1.8, label=r"$C^{\mathrm{bg-def}}_{0j}$")

    plt.axhline(0, color="black", lw=0.8)
    plt.axvline(0, color="gray", ls="--", lw=0.8, label="site 0")
    plt.axvline(-1, color="red", ls=":", lw=1.0, label="site 50")
    plt.axvline(1, color="orange", ls=":", lw=1.0, label="site 1")

    plt.xlabel("distance from site 0")
    plt.ylabel(r"real part")

    plt.title(
        rf"Signed cancellation in defect correction "
        rf"($N={N_def}$, $\lambda={lam}$, $\beta={beta}$)"
    )

    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()

    print("E0 =", data["E0"])
    print("f0 =", data["f0"])

    return data, x, delta, zero, bg


##sanity check for concureence eta relation and X form for rho 

def eta_sanity_check(
    N=51,
    beta=30.0,
    lam=1.0,
    pair=(0, 1),
    periodic=True,
    verbose=True
):
    eigvals, eigvecs, f, C = diagonal_data_from_hamiltonian(
        N=N, beta=beta, lam=lam, periodic=periodic
    )

    i, j = pair
    C_sub = C[[i, j]][:, [i, j]]

    a = C_sub[0, 0]
    b = C_sub[1, 1]
    g = C_sub[0, 1]

    eta = -2 * g

    rho_abg = rho_from_abg(a, b, g)
    conc_from_rho_abg = concurrence_general(rho_abg)

    conc_from_eta = concurrence_from_eta(eta)

    rho_direct = compute_rho(
        beta=beta,
        lam=lam,
        pair_idx=pair,
        N=N,
        periodic=periodic
    )
    conc_direct = concurrence_general(rho_direct)

    if verbose:
        print("==== ETA SANITY CHECK ====")
        print(f"N = {N}, beta = {beta}, lambda = {lam}, pair = {pair}")
        print()
        print("C_sub =")
        print(C_sub)
        print()
        print(f"C_ii = {a:.12f}")
        print(f"C_jj = {b:.12f}")
        print(f"C_ij = g = {g:.12f}")
        print(f"eta = -2g = {eta:.12f}")
        print()
        print(f"Concurrence from eta formula      = {conc_from_eta:.12f}")
        print(f"Concurrence from rho_from_abg     = {conc_from_rho_abg:.12f}")
        print(f"Concurrence from compute_rho      = {conc_direct:.12f}")
        print()
        print(f"|C_ii - 1/2| = {abs(a - 0.5):.3e}")
        print(f"|C_jj - 1/2| = {abs(b - 0.5):.3e}")

    return {
        "C_sub": C_sub,
        "a": a,
        "b": b,
        "g": g,
        "eta": eta,
        "conc_from_eta": conc_from_eta,
        "conc_from_rho_abg": conc_from_rho_abg,
        "conc_direct": conc_direct,
    }

#Entangement Phase Diagrams

def average_bell_from_rho(rho):
    """
    Horodecki CHSH maximum:
    B_max = 2 sqrt(m1 + m2),
    where m1,m2 are the two largest eigenvalues of T^T T.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sigmas = [sx, sy, sz]

    T = np.zeros((3, 3), dtype=float)

    for a in range(3):
        for b in range(3):
            op = np.kron(sigmas[a], sigmas[b])
            T[a, b] = np.real(np.trace(rho @ op))

    eigs = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
    return 2 * np.sqrt(eigs[0] + eigs[1])

def compute_pair_maps(
    selected_pairs=[(0, 1), (0, 50)],
    N=51,
    beta_vals=None,
    lambda_vals=None,
    periodic=True
):
    if beta_vals is None:
        beta_vals = np.geomspace(0.3, 50, 80)

    if lambda_vals is None:
        lambda_vals = np.linspace(-1, 1, 401)

    C_maps = {}
    B_maps = {}

    for pair in selected_pairs:
        C = np.zeros((len(beta_vals), len(lambda_vals)))
        B = np.zeros_like(C)

        for ib, beta in enumerate(beta_vals):
            for il, lam in enumerate(lambda_vals):
                rho = compute_rho(beta, lam, pair, N=N, periodic=periodic)
                C[ib, il] = concurrence_general(rho)
                B[ib, il] = average_bell_from_rho(rho)

        C_maps[pair] = C
        B_maps[pair] = B

    return beta_vals, lambda_vals, C_maps, B_maps

def plot_defect_phase_diagram_pair(
    pair,
    beta_vals,
    lambda_vals,
    C_map,
    B_map,
    eps_C=1e-4,
    bell_threshold=2.0,
    savepath=None
):
    phase = np.zeros_like(C_map)
    phase[C_map > eps_C] = 1              # entangled but Bell-local
    phase[B_map > bell_threshold] = 2     # Bell-violating

    fig, ax = plt.subplots(figsize=(7, 4.8))

    im = ax.pcolormesh(
        beta_vals,
        lambda_vals,
        phase.T,
        shading="auto"
    )

    ax.contour(beta_vals, lambda_vals, C_map.T, levels=[eps_C], linewidths=2)
    ax.contour(beta_vals, lambda_vals, B_map.T, levels=[bell_threshold], linewidths=2, linestyles="--")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(fr"Defect entanglement phase diagram, pair {pair}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["separable", "entangled/local", "Bell violating"])

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    plt.show()

def defect_pair_phase_diagram(
    pair=(0, 1),
    N=51,
    beta_vals=None,
    lambda_vals=None,
    periodic=True,
    eps_C=1e-4,
    bell_threshold=2.0,
    savepath=None
):
    if beta_vals is None:
        beta_vals = np.geomspace(0.3, 30, 80)

    if lambda_vals is None:
        lambda_vals = np.linspace(-1, 1, 301)

    C_map = np.zeros((len(beta_vals), len(lambda_vals)))
    B_map = np.zeros_like(C_map)

    for ib, beta in enumerate(beta_vals):
        for il, lam in enumerate(lambda_vals):
            rho = compute_rho(beta, lam, pair, N=N, periodic=periodic)
            C_map[ib, il] = concurrence_general(rho)
            B_map[ib, il] = average_bell_from_rho(rho)

    phase = np.zeros_like(C_map)
    phase[C_map > eps_C] = 1
    phase[B_map > bell_threshold] = 2

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.contourf(
        beta_vals,
        lambda_vals,
        phase.T,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=["lightgray", "khaki", "lightblue"],
        alpha=0.85
    )

    ax.contour(
        beta_vals,
        lambda_vals,
        C_map.T,
        levels=[eps_C],
        colors="red",
        linewidths=2
    )

    ax.contour(
        beta_vals,
        lambda_vals,
        B_map.T,
        levels=[bell_threshold],
        colors="blue",
        linewidths=2
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_ylim(lambda_vals[0], lambda_vals[-1])

    bc = "PBC" if periodic else "OBC"
    ax.set_title(fr"Defect Entanglement Phase Diagram, pair {pair} ({bc}, N={N})")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label=r"$\lambda_C(\beta)$, $C\to0$"),
        Line2D([0], [0], color="blue", lw=2, label=r"$\lambda_B(\beta)$, $\langle B\rangle \to 2$")
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=False)

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)

    plt.show()

    return beta_vals, lambda_vals, C_map, B_map, phase

def average_bell_from_rho(rho):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    sigmas = [sx, sy, sz]
    T = np.zeros((3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            T[i, j] = np.real(np.trace(rho @ np.kron(sigmas[i], sigmas[j])))

    eigs = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
    return 2 * np.sqrt(eigs[0] + eigs[1])

####concurrenceları aynı plota koyalım
def concurrence_boundary_for_pair(
    pair,
    N=51,
    beta_vals=None,
    lambda_vals=None,
    periodic=True,
    eps_C=1e-4
):
    if beta_vals is None:
        beta_vals = np.geomspace(0.3, 30, 80)

    if lambda_vals is None:
        lambda_vals = np.linspace(-1, 1, 401)

    lambda_c = []

    for beta in beta_vals:
        C_vals = []

        for lam in lambda_vals:
            rho = compute_rho(beta, lam, pair, N=N, periodic=periodic)
            C_vals.append(concurrence_general(rho))

        C_vals = np.array(C_vals)

        mask = C_vals > eps_C

        if not np.any(mask):
            lambda_c.append(np.nan)
            continue

        idx = np.where(mask)[0]

        # üst boundary: entangled bölgenin en büyük lambda sınırı
        j = idx[-1]
        lambda_c.append(lambda_vals[j])

    return beta_vals, np.array(lambda_c)

def plot_concurrence_boundaries_multiple_pairs(
    pairs=[(0, 1), (12, 13), (24, 25)],
    N=51,
    beta_vals=None,
    lambda_vals=None,
    periodic=True,
    eps_C=1e-4,
    savepath=None
):
    if beta_vals is None:
        beta_vals = np.geomspace(0.3, 30, 80)

    if lambda_vals is None:
        lambda_vals = np.linspace(-1, 1, 401)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for pair in pairs:
        beta_vals, lambda_c = concurrence_boundary_for_pair(
            pair=pair,
            N=N,
            beta_vals=beta_vals,
            lambda_vals=lambda_vals,
            periodic=periodic,
            eps_C=eps_C
        )

        ax.plot(beta_vals, lambda_c, lw=2, label=f"pair {pair}")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\lambda_C(\beta)$")
    ax.set_ylim(lambda_vals[0], lambda_vals[-1])

    bc = "PBC" if periodic else "OBC"
    ax.set_title(fr"Concurrence critical lines for selected pairs ({bc}, N={N})")

    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(frameon=False)

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)

    plt.show()

