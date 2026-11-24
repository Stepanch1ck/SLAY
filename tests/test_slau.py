"""
Модульные тесты для проверки корректности решения усечённой СЛАУ и упругих полей.
Поддерживает запуск как через unittest (стандартная библиотека), так и через pytest.
"""

import unittest
import numpy as np
from src.config import ProblemParameters
from src.slae_solver import SLAESolver, beta_jn, gamma_n, calculate_integrals_I_J
from src.fields import ElasticFields
from src.energy import EnergyFlux


class TestSLAE(unittest.TestCase):
    def test_beta_jn_branches(self):
        """Проверка правильности выбора ветвей квадратного корня beta_jn."""
        k = 2.0
        # 1. Затухающая мода (|L| >= k) -> чисто мнимая с Im > 0
        L_evanescent = 3.0
        b_evan = beta_jn(k, L_evanescent)
        self.assertAlmostEqual(b_evan.real, 0.0, places=7)
        self.assertGreater(b_evan.imag, 0.0)
        self.assertAlmostEqual(b_evan.imag, np.sqrt(3.0**2 - 2.0**2), places=7)

        # 2. Бегущая мода (|L| < k) -> отрицательная вещественная
        L_propagating = 1.0
        b_prop = beta_jn(k, L_propagating)
        self.assertAlmostEqual(b_prop.imag, 0.0, places=7)
        self.assertLess(b_prop.real, 0.0)
        self.assertAlmostEqual(b_prop.real, -np.sqrt(2.0**2 - 1.0**2), places=7)

    def test_integrals_I_and_J_properties(self):
        """Проверка фундаментальных свойств интегралов Ik и Jk."""
        l = 2.0 * np.pi
        a, b = 0.0, np.pi / 2.0
        k_arr = np.arange(-10, 11)

        I_dict, J_dict = calculate_integrals_I_J(k_arr, l, a, b)

        # I_0 = b - a
        self.assertAlmostEqual(I_dict[0].real, b - a, places=7)
        self.assertAlmostEqual(I_dict[0].imag, 0.0, places=7)

        # J_0 = l - I_0
        self.assertAlmostEqual(J_dict[0].real, l - (b - a), places=7)
        self.assertAlmostEqual(J_dict[0].imag, 0.0, places=7)

        # Для k != 0: J_k = -I_k
        for k in k_arr:
            if k != 0:
                self.assertAlmostEqual(J_dict[k].real, -I_dict[k].real, places=7)
                self.assertAlmostEqual(J_dict[k].imag, -I_dict[k].imag, places=7)

    def test_slae_dimensions_and_condition_number(self):
        """Проверка размерности матрицы и числа обусловленности СЛАУ при N=7."""
        params = ProblemParameters(N=7, rhs_mode="dissertation")
        solver = SLAESolver(params)
        res = solver.solve()

        self.assertEqual(res.A_matrix.shape, (15, 15))
        self.assertEqual(len(res.b_vector), 15)
        self.assertEqual(len(res.C_star), 15)
        self.assertEqual(len(res.C_n), 15)
        self.assertEqual(len(res.A_n), 15)
        self.assertLess(res.condition_number, 1e3)  # Хорошо обусловлена

    def test_boundary_condition_uy_zero(self):
        """Проверка тождественного обращения в ноль нормального перемещения uy_tot(x, 0)."""
        params = ProblemParameters(N=7)
        solver = SLAESolver(params)
        res = solver.solve()

        fields = ElasticFields(params, res.A_n, res.C_n)
        b_res = fields.verify_boundary_conditions(num_points=100)

        # uy_tot должно быть равно 0 с точностью до машинной погрешности (< 1e-12)
        self.assertLess(b_res.uy_tot_max, 1e-12)

    def test_single_element_rhs_mode(self):
        """Проверка режима правой части single_element (только элемент s отличен от нуля)."""
        params = ProblemParameters(N=3, s=0, rhs_mode="single_element")
        solver = SLAESolver(params)
        A, b, _, _, _, _ = solver.build_matrix_and_rhs()

        s_idx = params.s + params.N
        for i in range(len(b)):
            if i == s_idx:
                self.assertGreater(abs(b[i]), 0.0)
            else:
                self.assertAlmostEqual(abs(b[i]), 0.0, places=7)

    def test_energy_flux_positivity(self):
        """Проверка физического закона: уходящие волны переносят неотрицательную энергию."""
        params = ProblemParameters(N=7)
        solver = SLAESolver(params)
        res = solver.solve()

        energy = EnergyFlux(params, res.A_n, res.C_n).calculate()
        self.assertLess(energy.P_incident, 0.0)  # Падающая волна идет вниз
        self.assertGreaterEqual(energy.P_diffracted_total, 0.0)  # Дифрагированная волна уходит вверх


if __name__ == "__main__":
    unittest.main()
