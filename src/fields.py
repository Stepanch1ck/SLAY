"""
Модуль восстановления полей перемещений (ux, uy) и напряжений (sigma_x, sigma_y, tau).
Формулы (3.3) для падающей волны, (3.4) для дифрагированной волны и проверка граничных условий.
"""

from dataclasses import dataclass
import numpy as np
from .config import ProblemParameters
from .slae_solver import beta_jn


@dataclass
class FieldComponents:
    """Компоненты упругого поля в точках сетки (X, Y)."""
    ux: np.ndarray        # Перемещение по оси x
    uy: np.ndarray        # Перемещение по оси y
    sigma_x: np.ndarray   # Нормальное напряжение по оси x
    sigma_y: np.ndarray   # Нормальное напряжение по оси y
    tau: np.ndarray       # Касательное напряжение (напряжение сдвига)

    @property
    def displacement_magnitude(self) -> np.ndarray:
        """Модуль вектора перемещения |u| = sqrt(|ux|^2 + |uy|^2)."""
        return np.sqrt(np.abs(self.ux)**2 + np.abs(self.uy)**2)


@dataclass
class BoundaryVerificationResult:
    """Результаты проверки выполнения граничных условий при y = 0."""
    x_coords: np.ndarray
    uy_tot_max: float      # Должно быть ~ 0 на всем периоде [0, l]
    ux_rigid_max: float    # Должно быть мало на N (жесткий контакт)
    tau_defect_max: float  # Должно быть мало на M (свободное скольжение)
    uy_tot_l2: float
    ux_rigid_l2: float
    tau_defect_l2: float


class ElasticFields:
    """Класс для вычисления полей перемещений и напряжений в полуплоскости y >= 0."""

    def __init__(self, params: ProblemParameters, An: np.ndarray, Cn: np.ndarray):
        self.params = params
        self.An = np.asarray(An, dtype=complex)
        self.Cn = np.asarray(Cn, dtype=complex)
        self.n_vals = np.arange(-params.N, params.N + 1)

    def incident_fields(self, X: np.ndarray, Y: np.ndarray) -> FieldComponents:
        """
        Вычисление полей падающей волны (формула 3.3).
        X, Y могут быть 1D, 2D (meshgrid) или скалярами.
        """
        p = self.params
        Ls = p.L_n(p.s)
        b1s = beta_jn(p.k1, Ls)
        b2s = beta_jn(p.k2, Ls)

        phase_x = np.exp(1j * Ls * X)
        exp_y1 = np.exp(-1j * b1s * Y)
        exp_y2 = np.exp(-1j * b2s * Y)

        # Перемещения:
        # ux0 = (-Bs * Ls * exp(-i beta1s y) + Ds * beta2s * exp(-i beta2s y)) * exp(i Ls x)
        # uy0 = ( Bs * beta1s * exp(-i beta1s y) + Ds * Ls * exp(-i beta2s y)) * exp(i Ls x)
        ux = (-p.Bs * Ls * exp_y1 + p.Ds * b2s * exp_y2) * phase_x
        uy = (p.Bs * b1s * exp_y1 + p.Ds * Ls * exp_y2) * phase_x

        # Напряжения:
        # sigma_x = i * [ -Bs * (lambda k1^2 + 2 mu Ls^2) e^(-i beta1s y) + 2 Ds mu beta2s Ls e^(-i beta2s y) ] e^(i Ls x)
        # sigma_y = i * [ -Bs * (rho omega^2 - 2 mu Ls^2) e^(-i beta1s y) - 2 Ds mu beta2s Ls e^(-i beta2s y) ] e^(i Ls x)
        # tau = i * [ 2 Bs mu Ls beta1s e^(-i beta1s y) - Ds * (rho omega^2 - 2 mu Ls^2) e^(-i beta2s y) ] e^(i Ls x)
        lam_k1 = p.lambda_param * p.k1**2 + 2.0 * p.mu * Ls**2
        rho_om = p.rho * p.omega**2 - 2.0 * p.mu * Ls**2

        sig_x = 1j * (-p.Bs * lam_k1 * exp_y1 + 2.0 * p.Ds * p.mu * b2s * Ls * exp_y2) * phase_x
        sig_y = 1j * (-p.Bs * rho_om * exp_y1 - 2.0 * p.Ds * p.mu * b2s * Ls * exp_y2) * phase_x
        tau = 1j * (2.0 * p.Bs * p.mu * Ls * b1s * exp_y1 - p.Ds * rho_om * exp_y2) * phase_x

        return FieldComponents(ux=ux, uy=uy, sigma_x=sig_x, sigma_y=sig_y, tau=tau)

    def diffracted_fields(self, X: np.ndarray, Y: np.ndarray) -> FieldComponents:
        """
        Вычисление полей дифрагированной (уходящей) волны (формула 3.4).
        """
        p = self.params
        shape = np.broadcast(X, Y).shape
        ux = np.zeros(shape, dtype=complex)
        uy = np.zeros(shape, dtype=complex)
        sig_x = np.zeros(shape, dtype=complex)
        sig_y = np.zeros(shape, dtype=complex)
        tau = np.zeros(shape, dtype=complex)

        for idx, n in enumerate(self.n_vals):
            An = self.An[idx]
            Cn = self.Cn[idx]
            Ln = p.L_n(n)
            b1n = beta_jn(p.k1, Ln)
            b2n = beta_jn(p.k2, Ln)

            phase_x = np.exp(1j * Ln * X)
            exp_y1 = np.exp(1j * b1n * Y)
            exp_y2 = np.exp(1j * b2n * Y)

            # ux = [An * Ln * e^(i beta1n y) + Cn * beta2n * e^(i beta2n y)] * e^(i Ln x)
            # uy = [An * beta1n * e^(i beta1n y) - Cn * Ln * e^(i beta2n y)] * e^(i Ln x)
            ux += (An * Ln * exp_y1 + Cn * b2n * exp_y2) * phase_x
            uy += (An * b1n * exp_y1 - Cn * Ln * exp_y2) * phase_x

            # sigma_x = i * [ An * (lambda k1^2 + 2 mu Ln^2) e^(i beta1n y) + 2 Cn mu beta2n Ln e^(i beta2n y) ] * e^(i Ln x)
            # sigma_y = i * [ An * (rho omega^2 - 2 mu Ln^2) e^(i beta1n y) - 2 Cn mu beta2n Ln e^(i beta2n y) ] * e^(i Ln x)
            # tau = i * [ 2 An mu Ln beta1n e^(i beta1n y) + Cn * (rho omega^2 - 2 mu Ln^2) e^(i beta2n y) ] * e^(i Ln x)
            lam_k1 = p.lambda_param * p.k1**2 + 2.0 * p.mu * Ln**2
            rho_om = p.rho * p.omega**2 - 2.0 * p.mu * Ln**2

            sig_x += 1j * (An * lam_k1 * exp_y1 + 2.0 * Cn * p.mu * b2n * Ln * exp_y2) * phase_x
            sig_y += 1j * (An * rho_om * exp_y1 - 2.0 * Cn * p.mu * b2n * Ln * exp_y2) * phase_x
            tau += 1j * (2.0 * An * p.mu * Ln * b1n * exp_y1 + Cn * rho_om * exp_y2) * phase_x

        return FieldComponents(ux=ux, uy=uy, sigma_x=sig_x, sigma_y=sig_y, tau=tau)

    def total_fields(self, X: np.ndarray, Y: np.ndarray) -> FieldComponents:
        """Суммарное поле: падающая волна + дифрагированная волна."""
        f0 = self.incident_fields(X, Y)
        f_diff = self.diffracted_fields(X, Y)
        return FieldComponents(
            ux=f0.ux + f_diff.ux,
            uy=f0.uy + f_diff.uy,
            sigma_x=f0.sigma_x + f_diff.sigma_x,
            sigma_y=f0.sigma_y + f_diff.sigma_y,
            tau=f0.tau + f_diff.tau,
        )

    def verify_boundary_conditions(self, num_points: int = 500) -> BoundaryVerificationResult:
        """
        Проверка граничных условий на границе y = 0:
        1. uy_tot(x, 0) == 0 везде на [0, l]
        2. ux_tot(x, 0) == 0 на N (жесткий контакт)
        3. tau_tot(x, 0) == 0 на M (дефект)
        """
        p = self.params
        x = np.linspace(0.0, p.l_period, num_points, endpoint=False)
        y = np.zeros_like(x)

        tot = self.total_fields(x, y)

        # Маски для дефекта M и контакта N
        mask_M = (x > p.defect_a) & (x < p.defect_b)
        mask_N = (x >= p.defect_b) | (x <= p.defect_a)

        uy_tot_abs = np.abs(tot.uy)
        ux_rigid_abs = np.abs(tot.ux[mask_N]) if np.any(mask_N) else np.array([0.0])
        tau_defect_abs = np.abs(tot.tau[mask_M]) if np.any(mask_M) else np.array([0.0])

        return BoundaryVerificationResult(
            x_coords=x,
            uy_tot_max=float(np.max(uy_tot_abs)),
            ux_rigid_max=float(np.max(ux_rigid_abs)),
            tau_defect_max=float(np.max(tau_defect_abs)),
            uy_tot_l2=float(np.sqrt(np.mean(uy_tot_abs**2))),
            ux_rigid_l2=float(np.sqrt(np.mean(ux_rigid_abs**2))),
            tau_defect_l2=float(np.sqrt(np.mean(tau_defect_abs**2))),
        )
