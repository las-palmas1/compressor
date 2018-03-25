from scipy.interpolate import UnivariateSpline
import numpy as np
from abc import ABCMeta, abstractmethod
import typing
from scipy.interpolate import interp1d


class QuadraticCurve(metaclass=ABCMeta):
    def __init__(self, y1, y2, stage_num, num_fit_points: int=100, **kwargs):
        """
        :param y1: Значение величины в начале отрезка.
        :param y2: Значение величины в конце отрезка.
        :param stage_num: Число ступеней.
        :param num_fit_points: Число точек для получения кривой.
        :param kwargs: 1. angle1 и angle2 - углы касательных в точках конов кривой распределения;
                       2. y0 и x0 - координаты центрального полюса кривой (горизонтальная координата
                       задается в нормированном виде, т.е. в интервале от 0 до 1).

        Правило отсчета углов:
        Если половина касательной, лежащей вне интервала (x1, x2), находится ниже горизонтали проведенной из точки
        (x1, y1) или (x2, y2), то угол положительный, наоборот - отрицательный.11
        """
        self.y1 = y1
        self.y2 = y2
        self._angle1 = None
        self._angle2 = None
        self._y0 = None
        self._x0 = None
        if 'angle1' in kwargs and 'angle2' in kwargs:
            self._angle1 = kwargs['angle1']
            self._angle2 = kwargs['angle2']
            self._x0, self._y0 = self._get_central_pole(y1, 0, self._angle1, y2, 1, self._angle2)
        elif 'x0' in kwargs and 'y0' in kwargs:
            self._y0 = kwargs['y0']
            self._x0 = kwargs['x0']
            self._angle1, self._angle2 = self._get_angles(0, self.y1, 1, self.y2, self._x0, self._y0)
        self.num_fit_points = num_fit_points
        self.stage_num = stage_num

    @property
    def angle1(self):
        return self._angle1

    @angle1.setter
    def angle1(self, value):
        self._angle1 = value
        self._x0, self._y0 = self._get_central_pole(self.y1, 0, self._angle1, self.y2, 1, self._angle2)

    @property
    def angle2(self):
        return self._angle2

    @angle2.setter
    def angle2(self, value):
        self._angle2 = value
        self._x0, self._y0 = self._get_central_pole(self.y1, 0, self._angle1, self.y2, 1, self._angle2)

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value):
        self._x0 = value
        self._angle1, self._angle2 = self._get_angles(0, self.y1, 1, self.y2, self._x0, self._y0)

    @property
    def y0(self):
        return self._y0

    @y0.setter
    def y0(self, value):
        self._y0 = value
        self._angle1, self._angle2 = self._get_angles(0, self.y1, 1, self.y2, self._x0, self._y0)

    @classmethod
    def _get_angles(cls, x1, y1, x2, y2, x0, y0):
        angle1 = np.arctan((y0 - y1) / (x0 - x1))
        angle2 = np.arctan((y0 - y2) / (x2 - x0))
        return angle1, angle2

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

    @classmethod
    def get_array_from_dist(cls, dist: typing.Callable[[int], float], stage_num) -> typing.List[float]:
        res = [dist(i) for i in range(stage_num)]
        return res

    def get_array(self) -> typing.List[float]:
        x_arr_norm, y_arr = self._get_coordinates()
        x_arr = np.array(x_arr_norm) * self.stage_num
        dist = lambda x: interp1d(x_arr, y_arr)(x).__float__()
        return self.get_array_from_dist(dist, self.stage_num)


class QuadraticBezier(QuadraticCurve):
    def __init__(self, y1, y2, stage_num, num_fit_points: int=500, **kwargs):
        QuadraticCurve.__init__(self, y1, y2, stage_num, num_fit_points, **kwargs)

    @classmethod
    def _get_coordinate(cls, t, p1, p2, p3):
        return (1 - t)**2 * p1 + 2 * t * (1 - t) * p2 + t**2 * p3

    def _get_coordinates(self):
        t_arr = np.linspace(0, 1, self.num_fit_points)
        x_arr = [self._get_coordinate(t, 0, self._x0, 1) for t in t_arr]
        y_arr = [self._get_coordinate(t, self.y1, self._y0, self.y2) for t in t_arr]
        return x_arr, y_arr


class QuadraticSpline(QuadraticCurve):
    def __init__(self, y1, y2, stage_num, num_fit_points: int=500, **kwargs):
        QuadraticCurve.__init__(self, y1, y2, stage_num, num_fit_points, **kwargs)

    @classmethod
    def _get_spline(cls, x1, y1, x0, x2, y2, y0) -> UnivariateSpline:
        spline = UnivariateSpline([x1, x0, x2], [y1, y0, y2], k=2)
        return spline

    def _get_coordinates(self):
        x_arr = np.linspace(0, 1, self.num_fit_points)
        spline = self._get_spline(0, self.y1, self._x0, 1, self.y2, self._y0)
        y_arr = [spline(x) for x in x_arr]
        return x_arr, y_arr

