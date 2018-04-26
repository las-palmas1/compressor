from .stage_tools import Stage
import typing
from gas_turbine_cycle.gases import IdealGas
import numpy as np
from gas_turbine_cycle.tools.functions import eta_comp_stag_p
from gas_turbine_cycle.tools.gas_dynamics import GasDynamicFunctions as gd
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import logging


logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)


class Compressor:
    def __init__(
            self, work_fluid: IdealGas, stage_num: int, const_diam_par, p0_stag, T0_stag, G, n,
            H_t_rel_arr: typing.List[float],
            eta_ad_stag_arr: typing.List[float],
            R_av_arr: typing.List[float],
            k_h_arr: typing.List[float],
            c1_a_rel_arr: typing.List[float],
            h_rk_rel_arr: typing.List[float],
            h_na_rel_arr: typing.List[float],
            delta_a_rk_rel_arr: typing.List[float],
            delta_a_na_rel_arr: typing.List[float],
            d1_in_rel1, zeta_inlet, zeta_outlet, c11_init,
            precision=0.0001
    ):
        self.work_fluid = work_fluid
        self.stage_num = stage_num
        if 0 <= const_diam_par <= 1:
            self.const_diam_par = const_diam_par
        else:
            raise ValueError('const_diam_par should be greater than or equal to 0 and less than or equal to 1')
        self.p0_stag = p0_stag
        self.T0_stag = T0_stag
        self.G = G
        self.n = n
        self.H_t_rel_arr = H_t_rel_arr
        self.eta_ad_stag_arr = eta_ad_stag_arr
        self.R_av_arr = R_av_arr
        self.k_h_arr = k_h_arr
        self.c1_a_rel_arr = c1_a_rel_arr
        self.h_rk_rel_arr = h_rk_rel_arr
        self.h_na_rel_arr = h_na_rel_arr
        self.delta_a_rk_rel_arr = delta_a_rk_rel_arr
        self.delta_a_na_rel_arr = delta_a_na_rel_arr
        self.d1_in_rel1 = d1_in_rel1
        self.zeta_inlet = zeta_inlet
        self.zeta_outlet = zeta_outlet
        self.c11 = c11_init
        self.precision = precision
        self.c_p_av = self.work_fluid.c_p_real_func(self.T0_stag)
        self.k_av = self.work_fluid.k_func(self.c_p_av)
        self.k_av_old = None
        self.k_av_res = None
        self.c_p_av_old = None
        self.c11_old = None
        self.c11_res = None
        self.residual = None
        self._iter_num = None
        self._stages: typing.List[Stage] = None

    def __iter__(self):
        self._item = 0
        return self

    def __next__(self) -> Stage:
        if 0 <= self._item < self.stage_num:
            current = self._stages[self._item]
            self._item += 1
            return current
        else:
            raise StopIteration()

    def __getitem__(self, item):
        if 0 <= item < self.stage_num:
            return self._stages[item]
        else:
            raise IndexError('invalid index')

    def __len__(self):
        return self.stage_num

    def _get_stages(self) -> typing.List[Stage]:
        stages = []
        for i in range(self.stage_num):
            if i != self.stage_num - 1:
                stage = Stage(
                    k_av=self.k_av,
                    R_gas=self.work_fluid.R,
                    H_t_rel=self.H_t_rel_arr[i],
                    H_t_rel_next=self.H_t_rel_arr[i + 1],
                    u1_out=None,
                    k_h=self.k_h_arr[i],
                    eta_ad_stag=self.eta_ad_stag_arr[i],
                    d1_in_rel=None,
                    c1_a_rel=self.c1_a_rel_arr[i],
                    c3_a_rel=self.c1_a_rel_arr[i + 1],
                    R_av=self.R_av_arr[i],
                    R_av_next=self.R_av_arr[i + 1],
                    T1_stag=None,
                    p1_stag=None,
                    G=self.G,
                    n=self.n,
                    const_diam_par=self.const_diam_par,
                    precision=self.precision,
                    h_rk_rel=self.h_rk_rel_arr[i],
                    h_na_rel=self.h_na_rel_arr[i],
                    delta_a_rk_rel=self.delta_a_rk_rel_arr[i],
                    delta_a_na_rel=self.delta_a_na_rel_arr[i],
                )
            else:
                stage = Stage(
                    k_av=self.k_av,
                    R_gas=self.work_fluid.R,
                    H_t_rel=self.H_t_rel_arr[i],
                    H_t_rel_next=(self.H_t_rel_arr[i] - self.H_t_rel_arr[i - 1]) + self.H_t_rel_arr[i],
                    u1_out=None,
                    k_h=self.k_h_arr[i],
                    eta_ad_stag=self.eta_ad_stag_arr[i],
                    d1_in_rel=None,
                    c1_a_rel=self.c1_a_rel_arr[i],
                    c3_a_rel=(self.c1_a_rel_arr[i] - self.c1_a_rel_arr[i - 1]) + self.c1_a_rel_arr[i],
                    R_av=self.R_av_arr[i],
                    R_av_next=(self.R_av_arr[i] - self.R_av_arr[i - 1]) + self.R_av_arr[i],
                    T1_stag=None,
                    p1_stag=None,
                    G=self.G,
                    n=self.n,
                    const_diam_par=self.const_diam_par,
                    precision=self.precision,
                    h_rk_rel=self.h_rk_rel_arr[i],
                    h_na_rel=self.h_na_rel_arr[i],
                    delta_a_rk_rel=self.delta_a_rk_rel_arr[i],
                    delta_a_na_rel=self.delta_a_na_rel_arr[i],
                )
            stages.append(stage)
        return stages

    def _get_sigma_inlet(self, k, eps, lam):
        return 1 / (1 + self.zeta_inlet * (k / (k + 1)) * eps * lam**2)

    def _get_sigma_outlet(self, k, eps, lam):
        return 1 - self.zeta_outlet * (k / (k + 1)) * eps * lam**2

    def compute(self):
        self._stages = self._get_stages()
        self.work_fluid.T1 = self.T0_stag
        self.k_av_res = 1
        self.c11_res = 1
        self.residual = 1
        self._iter_num = 1
        while self.residual >= self.precision:
            logging.info('ITERATION %s' % self._iter_num)
            self.c_p_av_old = self.c_p_av
            self.c11_old = self.c11
            self.k_av_old = self.k_av
            self.c11, self.k_av, self.c_p_av = self._get_inlet_velocity(self.c11, self.k_av)
            self.k_av_res = abs(self.k_av - self.k_av_old) / self.k_av
            self.c11_res = abs(self.c11 - self.c11_old) / self.c11
            self.residual = max(self.k_av_res, self.c11_res)
            self._iter_num += 1
            logging.info('Residual = %.5f\n' % self.residual)
        self._compute_integrate_parameters()

    def _get_inlet_velocity(self, c11, k_av):
        self.a_cr11 = gd.a_cr(self.T0_stag, k_av, self.work_fluid.R)
        self.lam11 = c11 / self.a_cr11
        self.eps11 = gd.eps_lam(self.lam11, self.k_av)
        self.sigma_inlet = self._get_sigma_inlet(self.k_av, self.eps11, self.lam11)
        self._stages[0].p1_stag = self.p0_stag * self.sigma_inlet
        self._stages[0].T1_stag = self.T0_stag
        self._stages[0].geom.d1_in_rel = self.d1_in_rel1
        self.rho11 = self.p0_stag * self.sigma_inlet / (self.work_fluid.R * self.T0_stag) * self.eps11
        self.u1_out1 = (
                np.pi * self.G * self.n ** 2 / (900 * self.c1_a_rel_arr[0] * self.rho11 * (1 - self.d1_in_rel1 ** 2))
        ) ** (1 / 3)
        self._stages[0].u1_out = self.u1_out1
        for n, stage in enumerate(self._stages):
            stage.k_av = self.k_av
            logging.info('Computing stage %s' % (n + 1))
            if n < self.stage_num - 1:
                stage.compute()
                self._stages[n + 1].u1_out = stage.u3_out
                self._stages[n + 1].geom.d1_in_rel = stage.geom.d3_in_rel
                self._stages[n + 1].T1_stag = stage.T3_stag
                self._stages[n + 1].p1_stag = stage.p3_stag
            else:
                stage.compute()
        self.work_fluid.T2 = self.last.T3_stag
        return self._stages[0].c1, self.work_fluid.k_av_int, self.work_fluid.c_p_av_int

    @property
    def first(self) -> Stage:
        return self._stages[0]

    @property
    def last(self) -> Stage:
        return self._stages[self.stage_num - 1]

    def _compute_integrate_parameters(self):
        self.work_fluid.__init__()
        self.work_fluid.T1 = self.first.T1_stag
        self.work_fluid.T2 = self.last.T3_stag
        self.k_av = self.work_fluid.k_av_int
        self.c_p_av = self.work_fluid.c_p_av_int
        self.pi_la_stag = self.last.p3_stag / self.first.p1_stag
        self.eta_la_stag = self.T0_stag * (self.pi_la_stag ** ((self.k_av - 1) / self.k_av) - 1) / \
                           (self.last.T3_stag - self.first.T1_stag)
        self.sigma_outlet = self._get_sigma_outlet(self.last.k_av, gd.eps_lam(self.last.lam3, self.last.k_av),
                                                   self.last.lam3)
        self.pi_c_stag = self.pi_la_stag * self.sigma_inlet * self.sigma_outlet
        self.eta_c_stag = self.T0_stag * (self.pi_c_stag ** ((self.k_av - 1) / self.k_av) - 1) / \
                          (self.last.T3_stag - self.first.T1_stag)
        self.eta_c_stag_p = eta_comp_stag_p(self.pi_c_stag, self.k_av, self.eta_c_stag)

    def _get_r_arr(self):
        r_out_arr = []
        r_in_arr = []
        r_const_arr = []
        for n, stage in zip(range(self.stage_num), self):
            if n != self.stage_num - 1:
                r_out_arr.append(stage.geom.D1_out / 2)
                r_out_arr.append(stage.geom.D15_out / 2)
                r_out_arr.append(stage.geom.D20_out / 2)
                r_out_arr.append(stage.geom.D25_out / 2)
                r_in_arr.append(stage.geom.D1_in / 2)
                r_in_arr.append(stage.geom.D15_in / 2)
                r_in_arr.append(stage.geom.D20_in / 2)
                r_in_arr.append(stage.geom.D25_in / 2)
                r_const_arr.append(stage.geom.D_const / 2)
                r_const_arr.append(stage.geom.D_const / 2)
                r_const_arr.append(stage.geom.D_const / 2)
                r_const_arr.append(stage.geom.D_const / 2)
            else:
                r_out_arr.append(stage.geom.D1_out / 2)
                r_out_arr.append(stage.geom.D15_out / 2)
                r_out_arr.append(stage.geom.D20_out / 2)
                r_out_arr.append(stage.geom.D25_out / 2)
                r_out_arr.append(stage.geom.D3_out / 2)
                r_in_arr.append(stage.geom.D1_in / 2)
                r_in_arr.append(stage.geom.D15_in / 2)
                r_in_arr.append(stage.geom.D20_in / 2)
                r_in_arr.append(stage.geom.D25_in / 2)
                r_in_arr.append(stage.geom.D3_in / 2)
                r_const_arr.append(stage.geom.D_const / 2)
                r_const_arr.append(stage.geom.D_const / 2)
                r_const_arr.append(stage.geom.D_const / 2)
                r_const_arr.append(stage.geom.D_const / 2)
                r_const_arr.append(stage.geom.D_const / 2)
        return r_in_arr, r_const_arr, r_out_arr

    def _get_x_arr(self):
        x_arr = [0]
        x_current = 0
        for stage in self:
            x_arr.append(x_current + stage.geom.b_a_rk)
            x_arr.append(x_current + stage.geom.b_a_rk + stage.geom.delta_a_rk)
            x_arr.append(x_current + stage.geom.b_a_rk + stage.geom.delta_a_rk + stage.geom.b_a_na)
            x_arr.append(x_current + stage.geom.b_a_rk + stage.geom.delta_a_rk + stage.geom.b_a_na +
                         stage.geom.delta_a_na)
            x_current += stage.geom.length
        return x_arr

    def plot_geometry(self, figsize=(8, 6), fname: str=None):
        r_in_arr_fit, r_const_arr_fit, r_out_arr_fit = self._get_r_arr()
        x_arr_fit = self._get_x_arr()
        spline_out = InterpolatedUnivariateSpline(x_arr_fit, r_out_arr_fit)
        spline_av = InterpolatedUnivariateSpline(x_arr_fit, r_const_arr_fit)
        spline_in = InterpolatedUnivariateSpline(x_arr_fit, r_in_arr_fit)
        x_arr_new = np.linspace(0, x_arr_fit[len(x_arr_fit) - 1], 100)
        r_in_arr_new = [spline_in(x) for x in x_arr_new]
        r_const_arr_new = [spline_av(x) for x in x_arr_new]
        r_out_arr_new = [spline_out(x) for x in x_arr_new]

        plt.figure(figsize=figsize)
        plt.plot(x_arr_new, r_in_arr_new, lw=1.5, color='black')
        plt.plot(x_arr_new, r_const_arr_new, lw=0.7, color='black', ls='--')
        plt.plot(x_arr_new, r_out_arr_new, lw=1.5, color='black')

        for i in range(len(x_arr_fit)):
            if ((i + 1) / 2) % 2 == 0 or ((i + 1) / 2) % 2 == 1.5:
                color = 'crimson'
            else:
                color = 'darkblue'
            plt.plot([x_arr_fit[i], x_arr_fit[i]], [r_in_arr_fit[i], r_out_arr_fit[i]], lw=1.2, color=color)
        plt.grid()
        plt.ylim(ymin=0)
        plt.show()
        if fname:
            plt.savefig(fname)

    def _set_dist_plot(self, fname=None):
        plt.grid()
        plt.xlim(1, self.stage_num + 1)
        plt.xlabel(r'$z$', fontsize=12)
        x_arr = np.linspace(1, self.stage_num, self.stage_num)
        plt.xticks(x_arr, [int(x) for x in x_arr])
        plt.show()
        if fname:
            plt.savefig(fname)

    def plot_temp_dist(self, figsize=(7, 5), fname=None):
        plt.figure(figsize=figsize)
        x_arr = np.linspace(1, self.stage_num + 1, 2 * (self.stage_num + 1) - 1)
        y_arr = []
        for stage, n in zip(self, range(self.stage_num)):
            if n != self.stage_num - 1:
                y_arr.append(stage.T1_stag)
                y_arr.append(stage.T3_stag)
            else:
                y_arr.append(stage.T1_stag)
                y_arr.append(stage.T3_stag)
                y_arr.append(stage.T3_stag)
        plt.plot(x_arr, y_arr, lw=1.5, color='orange')
        plt.ylabel(r'$T^*,\ К$', fontsize=12)
        self._set_dist_plot(fname)

    def plot_press_dist(self, figsize=(7, 5), fname=None):
        plt.figure(figsize=figsize)
        x_arr = np.linspace(1, self.stage_num + 1, 2 * (self.stage_num + 1) - 1)
        y_arr = []
        for stage, n in zip(self, range(self.stage_num)):
            if n != self.stage_num - 1:
                y_arr.append(stage.p1_stag / 1e6)
                y_arr.append(stage.p3_stag / 1e6)
            else:
                y_arr.append(stage.p1_stag / 1e6)
                y_arr.append(stage.p3_stag / 1e6)
                y_arr.append(stage.p3_stag / 1e6)
        plt.plot(x_arr, y_arr, lw=1.5, color='orange')
        plt.ylabel(r'$p^*,\ МПа$', fontsize=12)
        self._set_dist_plot(fname)

    def plot_c_a_dist(self, figsize=(7, 5), fname=None):
        plt.figure(figsize=figsize)
        x_arr = np.linspace(1, self.stage_num + 1, 2 * (self.stage_num + 1) - 1)
        y_arr = []
        for stage, n in zip(self, range(self.stage_num)):
            if n != self.stage_num - 1:
                y_arr.append(stage.c1_a)
                y_arr.append(stage.c2_a)
            else:
                y_arr.append(stage.c1_a)
                y_arr.append(stage.c2_a)
                y_arr.append(stage.c3_a)
        plt.plot(x_arr, y_arr, lw=1.5, color='orange')
        plt.ylabel(r'$c_a,\ м/с$', fontsize=12)
        self._set_dist_plot(fname)

    def plot_init_param_dist(self, name: str, figsize=(7, 5), fname=None):
        get_attr = object.__getattribute__
        value_arr = get_attr(self, name + '_arr')
        x_arr = np.linspace(1, self.stage_num, self.stage_num)
        plt.figure(figsize=figsize)
        plt.plot(x_arr, value_arr, lw=1.5, color='orange', label=name)
        plt.legend(fontsize=12)
        if max(value_arr) <= 1:
            plt.ylim(ymin=0, ymax=1)
            plt.yticks(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
        else:
            plt.ylim(ymin=0)
        self._set_dist_plot(fname)











