"""
Модуль визуализации результатов расчета упругих полей и сходимости СЛАУ.
Построение пространственных распределений, граничных условий и спектров гармоник.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .config import ProblemParameters
from .fields import ElasticFields
from .slae_solver import SLAESolver


def plot_boundary_conditions(
    params: ProblemParameters,
    fields: ElasticFields,
    save_path: str | Path | None = None
) -> plt.Figure:
    """График распределения перемещений и напряжений вдоль границы раздела y = 0."""
    x = np.linspace(0.0, params.l_period, 400)
    y = np.zeros_like(x)

    tot = fields.total_fields(x, y)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1. Перемещение uy (должно быть 0 везде)
    axes[0].plot(x, np.abs(tot.uy), "b-", lw=2, label=r"$|u_y^{tot}(x, 0)|$ (нормаль)")
    axes[0].axvspan(params.defect_a, params.defect_b, color="orange", alpha=0.2, label=r"Дефект $\mathcal{M}$")
    axes[0].set_ylabel(r"$|u_y|$")
    axes[0].set_title(r"Проверка граничных условий при $y = 0$")
    axes[0].grid(True, ls="--", alpha=0.6)
    axes[0].legend(loc="upper right")

    # 2. Перемещение ux (должно быть 0 на N)
    axes[1].plot(x, np.abs(tot.ux), "r-", lw=2, label=r"$|u_x^{tot}(x, 0)|$ (тангенц.)")
    axes[1].axvspan(params.defect_a, params.defect_b, color="orange", alpha=0.2, label=r"Дефект $\mathcal{M}$ (скольжение)")
    axes[1].set_ylabel(r"$|u_x|$")
    axes[1].grid(True, ls="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    # 3. Касательное напряжение tau (должно быть 0 на M)
    axes[2].plot(x, np.abs(tot.tau), "g-", lw=2, label=r"$|\tau^{tot}(x, 0)|$ (сдвиг)")
    axes[2].axvspan(params.defect_a, params.defect_b, color="orange", alpha=0.2, label=r"Дефект $\mathcal{M}$ ($\tau = 0$)")
    axes[2].set_xlabel("Координата x")
    axes[2].set_ylabel(r"$|\tau|$")
    axes[2].grid(True, ls="--", alpha=0.6)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def plot_2d_field_maps(
    params: ProblemParameters,
    fields: ElasticFields,
    save_path: str | Path | None = None
) -> plt.Figure:
    """2D карты распределения модуля перемещений и компонент напряжений в полуплоскости y >= 0."""
    nx, ny = 120, 80
    x = np.linspace(0.0, params.l_period, nx)
    y = np.linspace(0.0, params.lambda_p, ny)
    X, Y = np.meshgrid(x, y)

    tot = fields.total_fields(X, Y)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # |u|
    im0 = axes[0, 0].pcolormesh(X, Y, tot.displacement_magnitude, shading="auto", cmap="viridis")
    axes[0, 0].set_title(r"Полное перемещение $|u(x, y)|$")
    fig.colorbar(im0, ax=axes[0, 0])

    # Re(ux)
    im1 = axes[0, 1].pcolormesh(X, Y, tot.ux.real, shading="auto", cmap="coolwarm")
    axes[0, 1].set_title(r"Горизонтальное перемещение $\mathrm{Re}(u_x)$")
    fig.colorbar(im1, ax=axes[0, 1])

    # Re(uy)
    im2 = axes[1, 0].pcolormesh(X, Y, tot.uy.real, shading="auto", cmap="coolwarm")
    axes[1, 0].set_title(r"Вертикальное перемещение $\mathrm{Re}(u_y)$")
    fig.colorbar(im2, ax=axes[1, 0])

    # |tau|
    im3 = axes[1, 1].pcolormesh(X, Y, np.abs(tot.tau), shading="auto", cmap="inferno")
    axes[1, 1].set_title(r"Касательное напряжение $|\tau(x, y)|$")
    fig.colorbar(im3, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # Отмечаем отрезок дефекта M на y = 0
        ax.plot([params.defect_a, params.defect_b], [0, 0], color="red", lw=4, label="Дефект M")

    axes[0, 0].legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def plot_coefficients_spectrum(
    n_vals: np.ndarray,
    An: np.ndarray,
    Cn: np.ndarray,
    save_path: str | Path | None = None
) -> plt.Figure:
    """Спектр амплитуд гармоник Флоке |An| и |Cn|."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.stem(n_vals - 0.15, np.abs(An), linefmt="b-", markerfmt="bo", basefmt="k-", label=r"$|A_n|$ (продольные)")
    ax.stem(n_vals + 0.15, np.abs(Cn), linefmt="r-", markerfmt="rs", basefmt="k-", label=r"$|C_n|$ (поперечные)")

    ax.set_xlabel("Номер гармоники n")
    ax.set_ylabel("Амплитуда")
    ax.set_title("Спектр комплексных коэффициентов дифрагированной волны")
    ax.set_xticks(n_vals)
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def plot_convergence_study(
    params_base: ProblemParameters,
    max_N: int = 15,
    save_path: str | Path | None = None
) -> plt.Figure:
    """Исследование сходимости коэффициентов C_s, C_{s-1}, C_{s+1} при N = 1..max_N."""
    N_list = list(range(1, max_N + 1))
    Cs_re, Cs_im = [], []
    Csp1_re, Csp1_im = [], []
    Csm1_re, Csm1_im = [], []

    for N_cur in N_list:
        p = ProblemParameters(
            vp=params_base.vp,
            vs=params_base.vs,
            rho=params_base.rho,
            omega=params_base.omega,
            l_period=params_base.l_period,
            defect_a=params_base.defect_a,
            defect_b=params_base.defect_b,
            s=params_base.s,
            theta=params_base.theta,
            angle_from_normal=params_base.angle_from_normal,
            alpha=params_base.alpha,
            Bs=params_base.Bs,
            Ds=params_base.Ds,
            N=N_cur,
            M=N_cur,
            rhs_mode=params_base.rhs_mode,
        )
        solver = SLAESolver(p)
        res = solver.solve()
        s_idx = p.s + N_cur

        Cs_val = res.C_n[s_idx]
        Cs_re.append(Cs_val.real)
        Cs_im.append(Cs_val.imag)

        if s_idx + 1 < len(res.C_n):
            Csp1_re.append(res.C_n[s_idx + 1].real)
            Csp1_im.append(res.C_n[s_idx + 1].imag)
        else:
            Csp1_re.append(np.nan)
            Csp1_im.append(np.nan)

        if s_idx - 1 >= 0:
            Csm1_re.append(res.C_n[s_idx - 1].real)
            Csm1_im.append(res.C_n[s_idx - 1].imag)
        else:
            Csm1_re.append(np.nan)
            Csm1_im.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Cs
    axes[0].plot(N_list, Cs_re, "b-o", label=r"$\mathrm{Re}(C_s)$")
    axes[0].plot(N_list, Cs_im, "b--s", label=r"$\mathrm{Im}(C_s)$")
    axes[0].set_xlabel("Порядок усечения N")
    axes[0].set_ylabel(r"$C_s$")
    axes[0].set_title(r"Сходимость $C_s$ от $N$")
    axes[0].grid(True, ls="--", alpha=0.6)
    axes[0].legend()

    # C_{s-1} and C_{s+1}
    axes[1].plot(N_list, Csp1_re, "r-o", label=r"$\mathrm{Re}(C_{s+1})$")
    axes[1].plot(N_list, Csp1_im, "r--s", label=r"$\mathrm{Im}(C_{s+1})$")
    axes[1].plot(N_list, Csm1_re, "m-^", label=r"$\mathrm{Re}(C_{s-1})$")
    axes[1].plot(N_list, Csm1_im, "m--v", label=r"$\mathrm{Im}(C_{s-1})$")
    axes[1].set_xlabel("Порядок усечения N")
    axes[1].set_ylabel(r"$C_{s\pm 1}$")
    axes[1].set_title(r"Сходимость высших гармоник от $N$")
    axes[1].grid(True, ls="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig
