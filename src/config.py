"""
Модуль конфигурации параметров задачи дифракции упругой волны Флоке.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ProblemParameters:
    """
    Класс параметров задачи дифракции упругой волны на границе полуплоскости
    с периодической системой дефектов (Глава 1 диссертации).

    Параметры среды:
        vp: скорость продольной волны (м/с или условные единицы)
        vs: скорость поперечной волны (м/с)
        rho: плотность среды (г/см^3 или кг/м^3)
        mu: модуль сдвига (Ляме)
        lambda_param: первый параметр Ляме
        omega: круговая частота (рад/с)

    Геометрия:
        l_period: период системы дефектов l вдоль оси Ox
        defect_a: начало интервала дефекта M (скольжение без трения)
        defect_b: конец интервала дефекта M, где 0 <= defect_a < defect_b <= l_period

    Падающая волна:
        s: номер возбуждающей гармоники Флоке (обычно s = 0)
        theta: угол падения волны в радианах (по умолчанию pi/4)
        angle_from_normal: True если угол theta отсчитывается от нормали к Ox,
                           False если угол отсчитывается от оси Ox (по умолчанию False)
        alpha: параметр Флоке (если None, рассчитывается автоматически через theta и k1)
        Bs: комплексная амплитуда продольной волны гармоники s
        Ds: комплексная амплитуда поперечной волны гармоники s

    Численные параметры:
        N: полуширина усечения СЛАУ (порядок системы 2N+1, гармоники от -N до N)
        M: полуширина суммирования внутреннего ряда по m (по умолчанию M = N)
        rhs_mode: 'dissertation' — точный вектор правой части формулы (3.15): b_k = D_tilde_s * I_{s-k}
                  'single_element' — вектор правой части, где отличен от нуля только элемент с индексом s
    """

    # Среда
    vp: float = 198.0
    vs: float = 54.0
    rho: float = 0.93
    omega: float = 150.0

    # Опционально можно переопределить lambda и mu
    mu: float = field(default=None)
    lambda_param: float = field(default=None)

    # Геометрия
    l_period: float = 2.0 * np.pi
    defect_a: float = 0.0
    defect_b: float = np.pi / 2.0

    # Падающая волна
    s: int = 0
    theta: float = np.pi / 4.0
    angle_from_normal: bool = False
    alpha: float = field(default=None)
    Bs: complex = 1.0 + 0.0j
    Ds: complex = 1.0 + 0.0j

    # Численные параметры
    N: int = 7
    M: int = field(default=None)
    rhs_mode: str = "dissertation"

    def __post_init__(self):
        # Расчет параметров Ляме
        if self.mu is None:
            self.mu = (self.vs ** 2) * self.rho
        if self.lambda_param is None:
            self.lambda_param = (self.vp ** 2) * self.rho - 2.0 * self.mu

        # Волновые числа k1 и k2 (формула 1.5)
        self.k1 = self.omega / self.vp
        self.k2 = self.omega / self.vs

        # Расчет параметра Флоке alpha, если не задан явно
        if self.alpha is None:
            if self.angle_from_normal:
                # Угол от нормали: alpha = k1 * sin(theta)
                self.alpha = self.k1 * np.sin(self.theta)
            else:
                # Угол от оси Ox: alpha = k1 * cos(theta)
                self.alpha = self.k1 * np.cos(self.theta)

        # По умолчанию M = N
        if self.M is None:
            self.M = self.N

        # Комплексные амплитуды
        self.Bs = complex(self.Bs)
        self.Ds = complex(self.Ds)

    @property
    def lambda_p(self) -> float:
        """Длина продольной падающей волны Lambda_p = 2*pi*v_p / omega."""
        return 2.0 * np.pi * self.vp / self.omega

    @property
    def lambda_s(self) -> float:
        """Длина поперечной падающей волны Lambda_s = 2*pi*v_s / omega."""
        return 2.0 * np.pi * self.vs / self.omega

    @property
    def matrix_dimension(self) -> int:
        """Размерность СЛАУ (2N+1)."""
        return 2 * self.N + 1

    def L_n(self, n: int | np.ndarray) -> float | np.ndarray:
        """Волновое число L_n = alpha + (2*pi/l) * n (формула 1.6)."""
        return self.alpha + (2.0 * np.pi / self.l_period) * n

    def summary(self) -> str:
        """Строковое описание ключевых параметров."""
        return (
            f"--- Параметры задачи ---\n"
            f"Среда: vp={self.vp} м/с, vs={self.vs} м/с, rho={self.rho}\n"
            f"Модули: mu={self.mu:.3e}, lambda={self.lambda_param:.3e}\n"
            f"Частота: omega={self.omega} рад/с, k1={self.k1:.4f}, k2={self.k2:.4f}\n"
            f"Длины волн: Lambda_p={self.lambda_p:.4f}, Lambda_s={self.lambda_s:.4f}, Период l={self.l_period:.4f}\n"
            f"Параметр Флоке alpha={self.alpha:.4f} (theta={np.degrees(self.theta):.2f}°)\n"
            f"Дефект M: ({self.defect_a:.4f}, {self.defect_b:.4f}) на отрезке [0, {self.l_period:.4f}]\n"
            f"Падающая волна: s={self.s}, Bs={self.Bs}, Ds={self.Ds}\n"
            f"Усечение: N={self.N} (размер матрицы {self.matrix_dimension}x{self.matrix_dimension}), M={self.M}\n"
            f"Режим правой части: {self.rhs_mode}"
        )
