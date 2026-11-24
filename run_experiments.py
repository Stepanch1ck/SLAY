"""
Скрипт для воспроизведения серии вычислительных экспериментов из диссертации
и запуска расчетов с параметрами из 'Исходные данные.txt'.
"""

import numpy as np
from src.config import ProblemParameters
from src.slae_solver import SLAESolver
from src.fields import ElasticFields
from src.energy import EnergyFlux


def experiment_from_initial_data_file():
    """Расчет с параметрами из файла 'Исходные данные.txt'."""
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ: ПАРАМЕТРЫ ИЗ 'Исходные данные.txt'")
    print("=" * 70)

    # Параметры из Исходные данные.txt:
    # l=1, Bs=2, Ds=3, s=1, ro=0.93, Vp=198, Vs=54, omega=1000, tetta=pi/3
    theta = np.pi / 3.0
    k1 = 1000.0 / 198.0
    alpha = k1 * np.cos(theta)

    params = ProblemParameters(
        vp=198.0,
        vs=54.0,
        rho=0.93,
        omega=1000.0,
        l_period=1.0,
        defect_a=0.0,
        defect_b=0.25,  # 1/4 периода
        s=1,
        theta=theta,
        alpha=alpha,
        Bs=2.0 + 0j,
        Ds=3.0 + 0j,
        N=7,
        M=7,
    )

    solver = SLAESolver(params)
    res = solver.solve()

    print(f"k1 = {params.k1:.4f}, k2 = {params.k2:.4f}, alpha = {params.alpha:.4f}")
    print(f"Размерность: {params.matrix_dimension}x{params.matrix_dimension}, cond(A) = {res.condition_number:.4e}")
    s_idx = params.s + params.N
    print(f"C_s (n = {params.s}): {res.C_n[s_idx]}")
    print(f"A_s (n = {params.s}): {res.A_n[s_idx]}")

    fields = ElasticFields(params, res.A_n, res.C_n)
    b_res = fields.verify_boundary_conditions()
    print(f"Неувязка uy_tot(x, 0): {b_res.uy_tot_max:.4e}")

    energy = EnergyFlux(params, res.A_n, res.C_n).calculate()
    print(f"Поток падающей волны P0: {energy.P_incident:.4e}, уходящей P: {energy.P_diffracted_total:.4e}")
    print(f"Баланс энергии: {energy.energy_balance_ratio * 100:.2f}%\n")


def experiment_plexiglas():
    """Расчет для органического стекла (плексигласа) при omega = 2800 (рис. 3.8 диссертации)."""
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ: ПЛЕКСИГЛАС (ПРИМЕР СТР. 24, РИС. 3.8)")
    print("=" * 70)

    params = ProblemParameters(
        vp=2672.0,
        vs=1284.0,
        rho=1.18,
        omega=2800.0,
        l_period=2.0 * np.pi,
        defect_a=0.0,
        defect_b=np.pi / 2.0,
        s=0,
        theta=np.pi / 4.0,
        Bs=1.0 + 0j,
        Ds=1.0 + 0j,
        N=7,
        M=7,
    )

    solver = SLAESolver(params)
    res = solver.solve()

    print(f"k1 = {params.k1:.4f}, k2 = {params.k2:.4f}, alpha = {params.alpha:.4f}")
    print(f"Размерность: {params.matrix_dimension}x{params.matrix_dimension}, cond(A) = {res.condition_number:.4e}")
    s_idx = params.s + params.N
    print(f"C_s (n = {params.s}): {res.C_n[s_idx]}")
    print(f"A_s (n = {params.s}): {res.A_n[s_idx]}")

    energy = EnergyFlux(params, res.A_n, res.C_n).calculate()
    print(f"Баланс энергии: {energy.energy_balance_ratio * 100:.2f}%\n")


if __name__ == "__main__":
    experiment_from_initial_data_file()
    experiment_plexiglas()
