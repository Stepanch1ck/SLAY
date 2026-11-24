"""
Модуль расчета энергетических характеристик упругих волн Флоке.
Формулы потока энергии через период l (раздел 2 диссертации, стр. 21-23).
Подготовка к Блоку 3 (индивидуальным заданиям преподавателя).
"""

from dataclasses import dataclass
import numpy as np
from .config import ProblemParameters
from .slae_solver import beta_jn


@dataclass
class EnergyFluxResult:
    """Результаты расчета потоков энергии."""
    P_incident: float              # Поток энергии падающей волны P^(0)
    P_diffracted_total: float      # Суммарный поток энергии дифрагированных волн P
    P_harmonics: dict[int, float]  # Потоки энергии отдельных гармоник P_n
    P_ratios: dict[int, float]     # Отношения P_n / |P^(0)|
    energy_balance_ratio: float    # Баланс энергии: P_diffracted_total / |P_incident|


class EnergyFlux:
    """Расчет плотности и суммарного потока энергии через период границы y = 0."""

    def __init__(self, params: ProblemParameters, An: np.ndarray, Cn: np.ndarray):
        self.params = params
        self.An = np.asarray(An, dtype=complex)
        self.Cn = np.asarray(Cn, dtype=complex)
        self.n_vals = np.arange(-params.N, params.N + 1)

    def calculate(self) -> EnergyFluxResult:
        """
        Вычисление потоков энергии по формулам теоремы 2.3 диссертации (стр. 23):
        Для уходящей волны:
            P = - (l/2) * rho * omega^3 * sum_n [ Re(beta_1n)*|An|^2 + Re(beta_2n)*|Cn|^2 ]
        Для падающей волны:
            P^(0) = (l/2) * rho * omega^3 * [ Re(beta_1s)*|Bs|^2 + Re(beta_2s)*|Ds|^2 ]
        """
        p = self.params
        factor = 0.5 * p.l_period * p.rho * (p.omega ** 3)

        # Падающая волна
        Ls = p.L_n(p.s)
        b1s = beta_jn(p.k1, Ls)
        b2s = beta_jn(p.k2, Ls)

        # Re(beta) отрицательно для бегущих волн, поэтому поток отрицателен (волна идет вниз)
        P0_signed = factor * (b1s.real * (abs(p.Bs)**2) + b2s.real * (abs(p.Ds)**2))
        P0_abs = abs(P0_signed)

        # Дифрагированные гармоники
        P_harmonics = {}
        P_ratios = {}
        P_diff_total = 0.0

        for idx, n in enumerate(self.n_vals):
            Ln = p.L_n(n)
            b1n = beta_jn(p.k1, Ln)
            b2n = beta_jn(p.k2, Ln)

            An = self.An[idx]
            Cn = self.Cn[idx]

            # Только моды с Re(beta) != 0 (бегущие волны) переносят энергию
            # Так как Re(beta) < 0, то -Re(beta) > 0 (уходящая вверх волна)
            Pn = -factor * (b1n.real * (abs(An)**2) + b2n.real * (abs(Cn)**2))
            P_harmonics[int(n)] = float(Pn)
            P_diff_total += Pn

            if P0_abs > 1e-14:
                P_ratios[int(n)] = float(Pn / P0_abs)
            else:
                P_ratios[int(n)] = 0.0

        balance = P_diff_total / P0_abs if P0_abs > 1e-14 else 0.0

        return EnergyFluxResult(
            P_incident=float(P0_signed),
            P_diffracted_total=float(P_diff_total),
            P_harmonics=P_harmonics,
            P_ratios=P_ratios,
            energy_balance_ratio=float(balance),
        )
