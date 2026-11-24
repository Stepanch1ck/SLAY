"""
Главный исполняемый скрипт для Блока 2 лабораторной работы:
Решение усечённой СЛАУ (формула 3.15) и восстановление упругих полей.
"""

from pathlib import Path
import numpy as np
from src.config import ProblemParameters
from src.slae_solver import SLAESolver
from src.fields import ElasticFields
from src.energy import EnergyFlux
from src.visualizer import (
    plot_boundary_conditions,
    plot_2d_field_maps,
    plot_coefficients_spectrum,
    plot_convergence_study,
)


def run_block2(
    N: int = 7,
    rhs_mode: str = "dissertation",
    save_plots: bool = True,
    output_dir: str = "output_plots"
):
    print("=" * 70)
    print("БЛОК 2. БАЗОВАЯ ЛАБОРАТОРНАЯ ЗАДАЧА: РЕШЕНИЕ УСЕЧЁННОЙ СЛАУ (3.15)")
    print("=" * 70)

    # 1. Задание входных параметров
    # Параметры из примера диссертации (Каучук, стр. 23)
    params = ProblemParameters(
        vp=198.0,
        vs=54.0,
        rho=0.93,
        omega=150.0,
        l_period=2.0 * np.pi,
        defect_a=0.0,
        defect_b=np.pi / 2.0,
        s=0,
        theta=np.pi / 4.0,
        Bs=1.0 + 0.0j,
        Ds=1.0 + 0.0j,
        N=N,
        rhs_mode=rhs_mode,
    )

    print("\n1. ВХОДНЫЕ ПАРАМЕТРЫ:")
    print(params.summary())

    # 2-4. Построение матрицы, вектора правой части и решение СЛАУ
    solver = SLAESolver(params)
    res = solver.solve()

    print("\n2. ХАРАКТЕРИСТИКИ СЛАУ A * X = b:")
    print(f"Размерность матрицы A: {params.matrix_dimension} x {params.matrix_dimension} (N = {params.N})")
    print(f"Число обусловленности cond(A): {res.condition_number:.4e}")
    print(f"Коэффициент D_tilde_s: {res.D_tilde_s:.6e}")
    print(f"Сдвиг для восстановления C_s (shift_s): {res.shift_s:.6e}")

    # 4-5. Таблица найденных коэффициентов C_n*, C_n, A_n
    print("\n4-5. РЕШЕНИЕ СЛАУ И ОБРАТНЫЙ ПЕРЕСЧЁТ КОЭФФИЦИЕНТОВ:")
    print("-" * 75)
    print(f"{'n':>4} | {'C_n* (Re, Im)':^22} | {'C_n (Re, Im)':^22} | {'A_n (Re, Im)':^22}")
    print("-" * 75)
    for idx, n in enumerate(res.n_indices):
        c_star_str = f"{res.C_star[idx].real:+.4e}, {res.C_star[idx].imag:+.4e}"
        cn_str = f"{res.C_n[idx].real:+.4e}, {res.C_n[idx].imag:+.4e}"
        an_str = f"{res.A_n[idx].real:+.4e}, {res.A_n[idx].imag:+.4e}"
        marker = " <== s" if n == params.s else ""
        print(f"{n:4d} | {c_star_str} | {cn_str} | {an_str}{marker}")
    print("-" * 75)

    # 6. Восстановление полей и проверка граничных условий
    fields = ElasticFields(params, res.A_n, res.C_n)
    b_res = fields.verify_boundary_conditions(num_points=500)

    print("\n6. ПРОВЕРКА ГРАНИЧНЫХ УСЛОВИЙ ПРИ y = 0:")
    print(f"  Макс. неувязка |u_y^tot| на [0, l] (должна быть 0): {b_res.uy_tot_max:.4e} (L2: {b_res.uy_tot_l2:.4e})")
    print(f"  Макс. неувязка |u_x^tot| на N (жесткий контакт):    {b_res.ux_rigid_max:.4e} (L2: {b_res.ux_rigid_l2:.4e})")
    print(f"  Макс. неувязка |tau^tot| на M (дефект, скольжение): {b_res.tau_defect_max:.4e} (L2: {b_res.tau_defect_l2:.4e})")

    # Энергетические характеристики (подготовка к Блоку 3)
    energy = EnergyFlux(params, res.A_n, res.C_n).calculate()
    print("\nЭНЕРГЕТИЧЕСКИЕ ХАРАКТЕРИСТИКИ (ГОТОВНОСТЬ К БЛОКУ 3):")
    print(f"  Поток энергии падающей волны P^(0):    {energy.P_incident:.4e}")
    print(f"  Суммарный поток уходящей волны P:      {energy.P_diffracted_total:.4e}")
    print(f"  Баланс энергии (P_diff / |P_inc|):     {energy.energy_balance_ratio * 100:.2f}%")
    print(f"  Отношение для гармоники s (P_s / P0):  {energy.P_ratios.get(params.s, 0.0):.4e}")

    # Визуализация
    if save_plots:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        f_bc = out_path / "boundary_conditions.png"
        plot_boundary_conditions(params, fields, save_path=f_bc)

        f_maps = out_path / "elastic_fields_2d.png"
        plot_2d_field_maps(params, fields, save_path=f_maps)

        f_spec = out_path / "coefficients_spectrum.png"
        plot_coefficients_spectrum(res.n_indices, res.A_n, res.C_n, save_path=f_spec)

        f_conv = out_path / "convergence_study.png"
        plot_convergence_study(params, max_N=15, save_path=f_conv)

        print(f"\nГрафики сохранены в директорию: {out_path.resolve()}")
        print(f"  - {f_bc.name} (проверка граничных условий при y=0)")
        print(f"  - {f_maps.name} (2D распределения полей ux, uy, tau)")
        print(f"  - {f_spec.name} (спектр гармоник An и Cn)")
        print(f"  - {f_conv.name} (сходимость коэффициентов от N)")

    print("\n" + "=" * 70)
    print("БЛОК 2 УСПЕШНО ВЫПОЛНЕН! РЕЗУЛЬТАТЫ ГОТОВЫ К СДАЧЕ.")
    print("=" * 70)
    return res, fields, energy


if __name__ == "__main__":
    run_block2(N=7)
