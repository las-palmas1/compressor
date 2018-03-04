from gas_turbine_cycle.gases import Air
from gas_turbine_cycle.tools.gas_dynamics import GasDynamicFunctions as gd
import numpy as np


class StageGeom:
    def __init__(self):
        self.F1 = None
        self.F2 = None
        self.const_diam_par = None
        self.D1_in = None
        self.D1_out = None
        self.D1_av = None
        self.D_const = None
        self.D2_in = None
        self.D2_out = None
        self.D2_av = None
        self.d1_in_rel = None
        self.d3_in_rel = None
        self.h_rk = None
        self.h_na = None
        self.b_a_rk = None
        self.b_a_na = None
        self.delta_a_rk = None
        self.delta_a_na = None
        self.delta_a_rk_rel = None
        self.delta_a_na_rel = None
        self.h_rk_rel = None
        self.h_na_rel = None

    def get_d_in_rel(self, F, const_diam_par, D_const):
        res = (
                      (np.pi / 4 * (1 + const_diam_par) * D_const ** 2 - F) /
                      (F * const_diam_par + np.pi / 4 * (1 + const_diam_par) * D_const ** 2)
        ) ** 0.5
        return res

    def get_D_const(self, d_in_rel, const_diam_par, D_out):
        res = D_out * np.sqrt((1 + d_in_rel**2 * const_diam_par) / (1 + const_diam_par))
        return res

    def get_D_out(self, F, d_in_rel):
        res = (4 * 1 / (np.pi * (1 - d_in_rel ** 2))) ** 0.5
        return res

    def compute_geom(self):
        self.h_rk = 0.5 * (self.D1_out - self.D1_in)
        self.b_a_rk = self.h_rk / self.h_rk_rel
        self.delta_a_rk = self.delta_a_rk_rel * self.h_rk


class Stage:
    def __init__(self, work_fluid: Air, H_t_rel, H_t_rel_next, u1_out, k_h, eta_ad_stag, d1_in_rel,
                 c1_a_rel, c3_a_rel, R_av, R_av_next, T1_stag, p1_stag, G, n,
                 const_diam_par=0.5, precision=0.001):
        """
        :param work_fluid: Рабочее тело.
        :param H_t_rel: Коэффициент напора ступени.
        :param H_t_rel_next: Коэффициент напора следующей ступени.
        :param u1_out: Окружная скорость на конце рабочей лопатки.
        :param k_h: Поправочный коэффициент для учета влияния вязкости робочего тела у втулки и корпуса.
        :param eta_ad_stag: Адиабатический КПД ступени.
        :param d1_in_rel: Относительный диаметр втулки на входе.
        :param c1_a_rel: Коэффициент расхода на входе в ступень.
        :param c3_a_rel: Коэффициент расхода на выходе из ступени.
        :param R_av: Степень реактивности на среднем радиусе ступени.
        :param R_av_next: Степень реактивности на среднем радиусе следующей ступени.
        :param T1_stag: Температура торможения на входе в ступень.
        :param p1_stag: Давление торможения на входе в ступень.
        :param G: Расход на входе в ступень.
        :param n: Частота вращения ротора компрессора, об/мин.
        :param const_diam_par: Параметр, определяющий положения постоянного диаметра ступени.
        :param precision: Точность итерационного расчета.
        """
        self.work_fluid = work_fluid
        self.H_t_rel = H_t_rel
        self.H_t_rel_next = H_t_rel_next
        self.u1_out = u1_out
        self.k_h = k_h
        self.eta_ad_stag = eta_ad_stag
        self.d1_in_rel = d1_in_rel
        self.c1_a_rel = c1_a_rel
        self.c1_a_rel_next = c3_a_rel
        self.R_av = R_av
        self.R_av_next = R_av_next
        self.T1_stag = T1_stag
        self.p1_stag = p1_stag
        self.G = G
        self.n = n
        self.const_diam_par = const_diam_par
        self.precision = precision
        self.c_p_av = None
        self.k_av = None
        self.R_gas = None
        self.k_av_old = None
        self.k_res = None

    def compute(self):
        self.work_fluid.T1 = self.T1_stag
        self.c_p_av = self.work_fluid.c_p_av_int
        self.k_av = self.work_fluid.k_av_int
        self.R_gas = self.work_fluid.R
        self.k_res = 1
        while self.k_res >= self.precision:
            self.k_av_old = self.k_av
            self._one_iter()
            self.work_fluid.T2 = self.T3_stag
            self.k_av = self.work_fluid.k_av_int
            self.c_p_av = self.work_fluid.c_p_av_int
            self.k_res = abs(self.k_av - self.k_av_old) / self.k_av

    def _one_iter(self):
        self.c1_a = self.u1_out * self.c1_a_rel
        self.H_t = self.H_t_rel * self.u1_out ** 2
        self.L_z = self.k_h * self.H_t
        self.H_ad = self.L_z * self.eta_ad_stag
        self.delta_T_stag = self.L_z / self.c_p_av
        self.T3_stag = self.T1_stag + self.delta_T_stag
        self.pi_stag = (1 + self.H_ad / (self.c_p_av * self.T1_stag)) ** (self.k_av / (self.k_av - 1))
        self.p3_stag = self.p1_stag * self.pi_stag
        self.a_cr1 = gd.a_cr(self.T1_stag, self.k_av, self.R_gas)
        self.a_cr3 = gd.a_cr(self.T3_stag, self.k_av, self.R_gas)
        self.r1_av_rel = ((1 + self.d1_in_rel) / 2) ** 0.5
        self.c1_u_rel = self.r1_av_rel * (1 - self.R_av) - self.H_t_rel / (2 * self.r1_av_rel)
        self.c1_u = self.c1_u_rel * self.u1_out
        self.c1 = np.sqrt(self.c1_u**2 + self.c1_a**2)
        self.alpha1 = np.arctan(self.c1_a_rel / self.c1_u_rel)
        self.lam1 = self.c1_a / (np.sin(self.alpha1) * self.a_cr1)
        self.q1 = gd.q(self.lam1, self.k_av)
        self.F1 = self.G * (self.R_gas * self.T1_stag) ** 0.5 / (gd.m(self.k_av) * self.p1_stag * self.q1 *
                                                                 np.sin(self.alpha1))
        self.D1_out = (4 * self.F1 / (np.pi * (1 - self.d1_in_rel ** 2))) ** 0.5
        self.D1_in = self.d1_in_rel * self.D1_out
        self.D_const = self.D1_out * np.sqrt((1 + self.d1_in_rel ** 2 * self.const_diam_par) / (1 + self.const_diam_par))
        self.alpha3 = self.alpha1
        self.u3_out = self.u1_out
        self.outlet_geom_res = 1
        while self.outlet_geom_res >= self.precision:
            self.alpha3_old = self.alpha3
            self.u3_out_old = self.u3_out
            self.alpha3, self.u3_out, self.outlet_geom_res = self._get_outlet_section_params(self.alpha3_old,
                                                                                             self.u3_out_old)
        self.r2_av_rel = 0.5 * (self.r1_av_rel + self.r3_av_rel)
        self.c2_u_rel = (self.H_t_rel + self.c1_u_rel * self.r1_av_rel) / self.r2_av_rel
        self.c2_a = 0.5 * (self.c1_a + self.c3_a)
        self.u2_out = 0.5 * (self.u1_out + self.u3_out)
        self.c2_a_rel = self.c2_a / self.u2_out
        self.beta1 = np.arctan(self.c1_a_rel / (self.r1_av_rel - self.c1_u_rel))
        self.beta2 = np.arctan(self.c2_a_rel / (self.r2_av_rel - self.c2_u_rel))
        self.alpha2 = np.arctan(self.c2_a_rel / self.c2_u_rel)
        self.epsilon_rk = self.beta2 - self.beta1
        self.epsilon_na = self.alpha3 - self.alpha2
        self.w1 = self.c1_a / np.sin(self.beta1)
        self.c2 = self.c2_a / np.sin(self.alpha2)
        self.tau1 = gd.tau_lam(self.lam1, self.k_av)
        self.T1 = self.T1_stag * self.tau1
        self.a1 = np.sqrt(self.k_av * self.R_gas * self.T1)
        self.M_w1_av = self.w1 / self.a1
        self.a_cr2 = self.a_cr3
        self.lam2 = self.c2 / self.a_cr2
        self.tau2 = gd.tau_lam(self.lam2, self.k_av)
        self.T2 = self.T3_stag * self.tau2
        self.a2 = np.sqrt(self.k_av * self.R_gas * self.T2)
        self.M_c2_av = self.c2 / self.a2

    def _get_outlet_section_params(self, alpha3, u3_out):
        self.c3_a = u3_out * self.c1_a_rel_next
        self.lam3 = self.c3_a / (np.sin(alpha3) * self.a_cr3)
        self.q3 = gd.q(self.lam3, self.k_av)
        self.F3 = self.F1 * (self.q1 * self.p1_stag) / (self.q3 * self.p3_stag) * np.sqrt(self.T3_stag / self.T1_stag)
        self.d3_in_rel = (
                (np.pi / 4 * (1 + self.const_diam_par) * self.D_const ** 2 - self.F3) /
                (self.F3 * self.const_diam_par + np.pi / 4 * (1 + self.const_diam_par) * self.D_const ** 2)
        ) ** 0.5
        self.r3_av_rel = ((1 + self.d3_in_rel ** 2) / 2) ** 0.5
        self.c3_u_rel = self.r3_av_rel * (1 - self.R_av_next) - self.H_t_rel_next / (2 * self.r3_av_rel)

        alpha3_new = np.arctan(self.c1_a_rel_next / self.c3_u_rel)
        self.D3_out = (4 * self.F3 / (np.pi * (1 - self.d3_in_rel ** 2)))
        u3_out_new = np.pi * self.D3_out * self.n / 60
        res = max(abs(alpha3 - alpha3_new) / alpha3, abs(u3_out - u3_out_new) / u3_out)
        return alpha3_new, u3_out_new, res








