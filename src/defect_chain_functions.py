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


#zero mode-bulk contribution plots

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

def semi_analytic_concurrence_pair(
    N=51,
    beta=30.0,
    lam=0.5,
    pair=(0, 1),
    periodic=True,
    use_actual_f0=True
):
    """
    Semi-analytic concurrence using:
        C_semi = C_full_odd_num - C_zero_num + C_zero_trial

    This avoids double-counting the zero-mode contribution.
    """

    i, j = pair

    eigvals, eigvecs, f, C_full = diagonal_data_from_hamiltonian(
        N=N,
        beta=beta,
        lam=lam,
        periodic=periodic
    )

    # numerical near-zero mode
    k0 = int(np.argmin(np.abs(eigvals)))
    E0 = eigvals[k0]
    psi0 = eigvecs[:, k0]

    # trial near-zero mode
    phi = trial_near_zero_profile(N=N, lam=lam, normalize=True)

    # align sign with numerical mode
    if np.dot(psi0, phi) < 0:
        phi = -phi

    f0 = f[k0] if use_actual_f0 else 0.5

    # remove numerical zero-mode contribution
    C_bg = C_full - f0 * np.outer(psi0, psi0)

    # add trial zero-mode contribution
    C_semi = C_bg + f0 * np.outer(phi, phi)

    C_sub = C_semi[[i, j]][:, [i, j]]

    a = C_sub[0, 0]
    b = C_sub[1, 1]
    g = C_sub[0, 1]

    rho = rho_from_abg(a, b, g)
    return concurrence_general(rho)

def plot_semi_analytic_concurrence_vs_lambda(
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
            (N-1, 0),
            (1, 2),
            (N-2, N-1),
            (N//4, N//4 + 1),
            (N - N//4 - 1, N - N//4),
        ]

    if savepath is None:
        savepath = FIG_DIR / f"semi_analytic_concurrence_N{N}.pdf"
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
                semi_analytic_concurrence_pair(
                    N=N,
                    beta=beta,
                    lam=lam,
                    pair=pair,
                    periodic=periodic
                )
            )

        ax.plot(lambda_vals, vals, label="semi-analytic")
        ax.set_title(f"pair {pair}")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Concurrence")
        ax.grid(True)

    for ax in axs[num:]:
        ax.axis("off")

    bc_type = "Periodic" if periodic else "Open"
    fig.suptitle(
        rf"Semi-analytic concurrence vs $\lambda$ ($N={N}$, $\beta={beta}$, {bc_type})",
        fontsize=14
    )

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    fig.savefig(savepath, bbox_inches="tight")
    plt.show()

