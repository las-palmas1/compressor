from scipy.interpolate import UnivariateSpline
import numpy as np
from abc import ABCMeta, abstractmethod
import typing
from scipy.interpolate import interp1d


class QuadraticCurve(metaclass=ABCMeta):
    def __init__(self, y1, angle1, y2, angle2, stage_num, num_fit_points: int=100):
        self.y1 = y1
        self.angle1 = angle1
        self.y2 = y2
        self.angle2 = angle2
        self.num_fit_points = num_fit_points
        self.stage_num = stage_num
        self.x0, self.y0 = self._get_central_pole(y1, 0, angle1, y2, 1, angle2)

    @classmethod
    def _get_central_pole(cls, y1, x1, angle1, y2, x2, angle2):
        """
        :param y1:
        :param x1:
        :param angle1: Угол наклона к горизонтале касательной к сплайну на входе, рад.
        :param y2: Угол наклона к горизонтале касательной к сплайну на выходе, рад.
        :param x2:
        :param angle2:
        :return:
        Возвращает координаты центрального полюса сплайна. Углы отсчитываются от горизонтали.
        Правило отсчета углов:
        Если половина касательной, лежащей вне интервала (x1, x2), находится ниже горизонтали проведенной из точки
        (x1, y1) или (x2, y2), то угол положительный, наоборот - отрицательный.
        """
        x0 = (y2 - y1 + x2 * np.tan(angle2) + x1 * np.tan(angle1)) / (np.tan(angle1) + np.tan(angle2))
        y0 = y1 + (x0 - x1) * np.tan(angle1)
        return x0, y0

    @abstractmethod
    def _get_coordinates(self) -> tuple:
        pass

    def get_dist(self) -> typing.Callable[[int], float]:
        x_arr_norm, y_arr = self._get_coordinates()
        x_arr = np.array(x_arr_norm) * self.stage_num
        return lambda x: interp1d(x_arr, y_arr)(x)


class QuadraticBezier(QuadraticCurve):
    def __init__(self, y1, angle1, y2, angle2, stage_num, num_fit_points: int=500):
        QuadraticCurve.__init__(self, y1, angle1, y2, angle2, stage_num, num_fit_points)

    @classmethod
    def _get_coordinate(cls, t, p1, p2, p3):
        return (1 - t)**2 * p1 + 2 * t * (1 - t) * p2 + t**2 * p3

    def _get_coordinates(self):
        t_arr = np.linspace(0, 1, self.num_fit_points)
        x_arr = [self._get_coordinate(t, 0, self.x0, 1) for t in t_arr]
        y_arr = [self._get_coordinate(t, self.y1, self.y0, self.y2) for t in t_arr]
        return x_arr, y_arr


class QuadraticSpline(QuadraticCurve):
    def __init__(self, y1, angle1, y2, angle2, stage_num, num_fit_points: int=500):
        QuadraticCurve.__init__(self, y1, angle1, y2, angle2, stage_num, num_fit_points)

    @classmethod
    def _get_spline(cls, x1, y1, angle1, x2, y2, angle2) -> UnivariateSpline:
        x0, y0 = cls._get_central_pole(y1, x1, angle1, y2, x2, angle2)
        spline = UnivariateSpline([x1, x0, x2], [y1, y0, y2], k=2)
        return spline

    def _get_coordinates(self):
        x_arr = np.linspace(0, 1, self.num_fit_points)
        spline = self._get_spline(0, self.y1, self.angle1, 1, self.y2, self.angle2)
        y_arr = [spline(x) for x in x_arr]
        return x_arr, y_arr

