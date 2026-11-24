"""
Пакет для моделирования дифракции упругих волн на периодической системе дефектов.
Решение усечённой СЛАУ (формула 3.15 диссертации / ММЭТ).
"""

from .config import ProblemParameters
from .slae_solver import SLAESolver, beta_jn, gamma_n, calculate_integrals_I_J
from .fields import ElasticFields
from .energy import EnergyFlux

__all__ = [
    "ProblemParameters",
    "SLAESolver",
    "beta_jn",
    "gamma_n",
    "calculate_integrals_I_J",
    "ElasticFields",
    "EnergyFlux",
]
