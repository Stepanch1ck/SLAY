"""
Модуль решения усечённой СЛАУ (формула 3.15).
Формирование матрицы коэффициентов, расчет интегралов Ik и Jk,
нахождение комплексных коэффициентов C_n* (C_tilde) и обратный пересчет в Cn и An.
"""

from dataclasses import dataclass
import numpy as np
from .config import ProblemParameters


def beta_jn(k: float, L: float | np.ndarray) -> complex | np.ndarray:
    """
    Вычисление корня beta_jn = sqrt(k^2 - L^2) по формуле (п. 1.3 диссертации):
      beta_jn = i * sqrt(L^2 - k^2),   при |L| >= k (затухающие моды)
      beta_jn = -sqrt(k^2 - L^2),     при |L| < k  (бегущие моды в направлении +y)
    """
    L_arr = np.asarray(L, dtype=float)
    is_scalar = (L_arr.ndim == 0)
    L_flat = np.atleast_1d(L_arr)

    cond = np.abs(L_flat) >= k
    res = np.where(
        cond,
        1j * np.sqrt(np.maximum(0.0, L_flat**2 - k**2)),
        -np.sqrt(np.maximum(0.0, k**2 - L_flat**2))
    )
    return res[0] if is_scalar else res


def gamma_n(params: ProblemParameters, n: int | np.ndarray) -> complex | np.ndarray:
    """
    Коэффициент гамма_n = L_n^2 / beta_1n + beta_2n (формула 3.10).
    """
    L = params.L_n(n)
    b1 = beta_jn(params.k1, L)
    b2 = beta_jn(params.k2, L)
    return (L**2) / b1 + b2


def calculate_integrals_I_J(
    k_array: np.ndarray,
    l_period: float,
    defect_a: float,
    defect_b: float
) -> tuple[dict[int, complex], dict[int, complex]]:
    """
    Вычисление интегралов I_k и J_k (формула 3.12):
        I_k = int_{M} exp(i * 2*pi*k*t / l) dt
        J_k = int_{N} exp(i * 2*pi*k*t / l) dt

    Для дефекта M = (a, b) на отрезке [0, l]:
        k = 0:
            I_0 = b - a
            J_0 = l - I_0
        k != 0:
            I_k = [exp(i * 2*pi*k*b / l) - exp(i * 2*pi*k*a / l)] / (i * 2*pi*k / l)
            J_k = -I_k (так как int_0^l exp(i * 2*pi*k*t / l) dt = 0)
    """
    I_dict: dict[int, complex] = {}
    J_dict: dict[int, complex] = {}

    for k in k_array:
        k_int = int(k)
        if k_int == 0:
            I_val = complex(defect_b - defect_a, 0.0)
            J_val = complex(l_period - (defect_b - defect_a), 0.0)
        else:
            denom = 1j * 2.0 * np.pi * k_int / l_period
            exp_b = np.exp(1j * 2.0 * np.pi * k_int * defect_b / l_period)
            exp_a = np.exp(1j * 2.0 * np.pi * k_int * defect_a / l_period)
            I_val = (exp_b - exp_a) / denom
            J_val = -I_val

        I_dict[k_int] = I_val
        J_dict[k_int] = J_val

    return I_dict, J_dict


@dataclass
class SLAEResult:
    """Результаты решения СЛАУ и обратного пересчета."""
    n_indices: np.ndarray        # Массив индексов гармоник [-N, ..., N]
    A_matrix: np.ndarray         # Матрица коэффициентов (2N+1) x (2N+1)
    b_vector: np.ndarray         # Вектор правой части (2N+1)
    C_star: np.ndarray           # Решение СЛАУ C_n* (в диссертации tilde{C}_n)
    C_n: np.ndarray              # Восстановленный вектор C_n
    A_n: np.ndarray              # Восстановленный вектор A_n
    condition_number: float      # Число обусловленности матрицы A
    D_tilde_s: complex           # Величина D_tilde_s
    shift_s: complex             # Добавка для восстановления C_s


class SLAESolver:
    """Решатель усечённой системы линейных уравнений (формула 3.15)."""

    def __init__(self, params: ProblemParameters):
        self.params = params

    def build_matrix_and_rhs(self) -> tuple[np.ndarray, np.ndarray, dict, dict, complex, complex]:
        """
        Формирование матрицы коэффициентов A размера (2N+1)x(2N+1) с учетом главной диагонали
        и вектора правой части b (формула 3.15).
        """
        p = self.params
        N = p.N
        M = p.M
        dim = 2 * N + 1
        n_vals = np.arange(-N, N + 1)
        m_vals = np.arange(-M, M + 1)

        # Вычисляем вспомогательные величины для гармоники s
        Ls = p.L_n(p.s)
        b1s = beta_jn(p.k1, Ls)
        b2s = beta_jn(p.k2, Ls)
        denom_s = Ls**2 + b1s * b2s

        # D_tilde_s (формула на стр. 16/27)
        D_tilde_s = 2.0 * Ls * (p.Bs * b1s + p.Ds) / denom_s

        # Сдвиг для обратного пересчета C_s (формула на стр. 16/27):
        # C_s = C_tilde_s + (2*Bs*Ls*b1s + Ds*(Ls^2 - b1s*b2s)) / (Ls^2 + b1s*b2s)
        shift_s = (2.0 * p.Bs * Ls * b1s + p.Ds * (Ls**2 - b1s * b2s)) / denom_s

        # Предварительный расчет гамма_n и гамма_m
        gamma_n_vals = gamma_n(p, n_vals)
        gamma_m_vals = gamma_n(p, m_vals)

        # Предварительный расчет интегралов I_k и J_k
        # Разности (n - m) лежат в диапазоне [-N - M, N + M]
        # Разности (m - k) лежат в диапазоне [-M - N, M + N]
        # Разности (s - k) лежат в диапазоне [s - N, s + N]
        max_diff = N + M + abs(p.s) + 5
        diff_array = np.arange(-max_diff, max_diff + 1)
        I_dict, J_dict = calculate_integrals_I_J(diff_array, p.l_period, p.defect_a, p.defect_b)

        # Формирование матрицы A и вектора b
        A = np.zeros((dim, dim), dtype=complex)
        b = np.zeros(dim, dtype=complex)

        for row_idx, k in enumerate(n_vals):
            # Вектор правой части:
            if p.rhs_mode == "single_element":
                # Вектор справа отличен от нуля только в элементе с индексом s
                if k == p.s:
                    b[row_idx] = D_tilde_s * I_dict[0]
                else:
                    b[row_idx] = 0.0 + 0.0j
            else:
                # Стандартный режим диссертации (формула 3.15): b_k = D_tilde_s * I_{s-k}
                b[row_idx] = D_tilde_s * I_dict[p.s - k]

            for col_idx, n in enumerate(n_vals):
                # Внутренняя сумма по m от -M до M: sum_m (1 / gamma_m) * I_{n-m} * J_{m-k}
                inner_sum = 0.0 + 0.0j
                for m_idx, m in enumerate(m_vals):
                    inv_gamma_m = 1.0 / gamma_m_vals[m_idx]
                    inner_sum += inv_gamma_m * I_dict[n - m] * J_dict[m - k]

                # Элемент главной диагонали l * delta_{k,n}
                delta_kn = 1.0 if k == n else 0.0
                A[row_idx, col_idx] = p.l_period * delta_kn - (1.0 / p.l_period) * gamma_n_vals[col_idx] * inner_sum

        return A, b, I_dict, J_dict, D_tilde_s, shift_s

    def solve(self) -> SLAEResult:
        """
        Решает усечённую СЛАУ A * X = b, находит C_n* (от -N до N),
        и производит обратный пересчет в C_n и A_n.
        """
        p = self.params
        A, b, _, _, D_tilde_s, shift_s = self.build_matrix_and_rhs()

        # Число обусловленности
        cond_A = float(np.linalg.cond(A))

        # Решение СЛАУ методом линейной алгебры (Gauss / LU-факторизация)
        C_star = np.linalg.solve(A, b)

        # Обратный пересчет в C_n:
        # C_n = C_star_n при n != s
        # C_s = C_star_s + shift_s при n == s
        C_n = C_star.copy()
        s_idx = p.s + p.N
        if 0 <= s_idx < len(C_n):
            C_n[s_idx] += shift_s

        # Обратный пересчет в A_n (формула 3.7):
        # A_n = (L_n / beta_1n) * C_n при n != s
        # A_s = (L_s / beta_1s) * (C_s - D_s) - B_s при n == s
        A_n = np.zeros_like(C_n, dtype=complex)
        n_vals = np.arange(-p.N, p.N + 1)

        Ls = p.L_n(p.s)
        b1s = beta_jn(p.k1, Ls)

        for idx, n in enumerate(n_vals):
            Ln = p.L_n(n)
            b1n = beta_jn(p.k1, Ln)
            if n == p.s:
                A_n[idx] = (Ls / b1s) * (C_n[idx] - p.Ds) - p.Bs
            else:
                A_n[idx] = (Ln / b1n) * C_n[idx]

        return SLAEResult(
            n_indices=n_vals,
            A_matrix=A,
            b_vector=b,
            C_star=C_star,
            C_n=C_n,
            A_n=A_n,
            condition_number=cond_A,
            D_tilde_s=D_tilde_s,
            shift_s=shift_s,
        )
