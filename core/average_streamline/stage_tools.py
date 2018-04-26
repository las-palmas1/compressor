from gas_turbine_cycle.tools.gas_dynamics import GasDynamicFunctions as gd
import numpy as np
import matplotlib.pyplot as plt


class StageGeom:
    """
    Индекс 1 - сечение на входе в РК, 3 - выходе из ступени, 15 - выходе из РК, 20 - входе в НА,
    25 - выходе из НА.
    """
    def __init__(self):
        self.F1 = None
        self.F3 = None
        self.const_diam_par = None
        self.D1_in = None
        self.D1_out = None
        self.D1_av = None
        self.D_const = None
        self.D3_in = None
        self.D3_out = None
        self.D3_av = None
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
        self.length = None
        self.F15 = None
        self.F20 = None
        self.F25 = None
        self.D15_in = None
        self.D20_in = None
        self.D25_in = None
        self.D15_out = None
        self.D20_out = None
        self.D25_out = None
        self.d15_in_rel = None
        self.d20_in_rel = None
        self.d25_in_rel = None
        self.r1_av_rel = None
        self.r2_av_rel = None
        self.r3_av_rel = None

    @classmethod
    def _get_out_in_ring_ratio(cls, const_diam_par):
        if const_diam_par != 0:
            return (1 - const_diam_par) / const_diam_par
        else:
            return 1e4

    @classmethod
    def get_d_in_rel(cls, F, const_diam_par, D_const):
        out_in_ring_ratio = cls._get_out_in_ring_ratio(const_diam_par)
        res = (
                      (np.pi / 4 * (1 + out_in_ring_ratio) * D_const ** 2 - F) /
                      (F * out_in_ring_ratio + np.pi / 4 * (1 + out_in_ring_ratio) * D_const ** 2)
        ) ** 0.5
        return res

    @classmethod
    def get_D_const(cls, d_in_rel, const_diam_par, D_out):
        out_in_ring_ratio = cls._get_out_in_ring_ratio(const_diam_par)
        res = D_out * np.sqrt((1 + d_in_rel**2 * out_in_ring_ratio) / (1 + out_in_ring_ratio))
        return res

    @classmethod
    def get_D_out(cls, F, d_in_rel):
        res = (4 * F / (np.pi * (1 - d_in_rel ** 2))) ** 0.5
        return res

    @classmethod
    def get_F(cls, x, length, F1, F2):
        return F1 + (F2 - F1) / length * x

    def compute(self):
        self.h_rk = 0.5 * (self.D1_out - self.D1_in)
        self.b_a_rk = self.h_rk / self.h_rk_rel
        self.delta_a_rk = self.delta_a_rk_rel * self.h_rk
        self.h_na = 0.5 * (self.D3_out - self.D3_in)
        self.b_a_na = self.h_na / self.h_na_rel
        self.delta_a_na = self.delta_a_na_rel * self.h_na
        self.length = self.b_a_rk + self.b_a_na + self.delta_a_rk + self.delta_a_na
        self.F15 = self.get_F(self.b_a_rk, self.length, self.F1, self.F3)
        self.F20 = self.get_F(self.b_a_rk + self.delta_a_rk, self.length, self.F1, self.F3)
        self.F25 = self.get_F(self.b_a_rk + self.delta_a_rk + self.b_a_na, self.length, self.F1, self.F3)
        self.d15_in_rel = self.get_d_in_rel(self.F15, self.const_diam_par, self.D_const)
        self.d20_in_rel = self.get_d_in_rel(self.F20, self.const_diam_par, self.D_const)
        self.d25_in_rel = self.get_d_in_rel(self.F25, self.const_diam_par, self.D_const)
        self.D15_out = self.get_D_out(self.F15, self.d15_in_rel)
        self.D20_out = self.get_D_out(self.F20, self.d20_in_rel)
        self.D25_out = self.get_D_out(self.F25, self.d25_in_rel)
        self.D15_in = self.D15_out * self.d15_in_rel
        self.D20_in = self.D20_out * self.d20_in_rel
        self.D25_in = self.D25_out * self.d25_in_rel
        self.D1_av = np.sqrt(self.D1_out**2 - 2 * self.F1 / np.pi)
        self.D3_av = np.sqrt(self.D3_out**2 - 2 * self.F1 / np.pi)


class Stage:
    def __init__(self, k_av, R_gas, H_t_rel, H_t_rel_next, u1_out, k_h, eta_ad_stag, d1_in_rel,
                 c1_a_rel, c3_a_rel, R_av, R_av_next, T1_stag, p1_stag, G, n, h_rk_rel=2.5, h_na_rel=3.0,
                 delta_a_rk_rel=0.3, delta_a_na_rel=0.3, const_diam_par=0.5, precision=0.001):
        """
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
        :param h_rk_rel: Отношение высоты РК к осевой ширине.
        :param h_na_rel: Отношение высоты НА к осевой ширине.
        :param delta_a_rk_rel: Отношение осевого зазора после РК к выооте РК.
        :param delta_a_na_rel: Отношение осевого зазора после НА к выооте НА.
        :param const_diam_par: Параметр, определяющий положения постоянного диаметра ступени.
        :param precision: Точность итерационного расчета.
        """
        self.k_av = k_av
        self.R_gas = R_gas
        self.H_t_rel = H_t_rel
        self.H_t_rel_next = H_t_rel_next
        self.u1_out = u1_out
        self.k_h = k_h
        self.eta_ad_stag = eta_ad_stag
        self.c1_a_rel = c1_a_rel
        self.c3_a_rel = c3_a_rel
        self.R_av = R_av
        self.R_av_next = R_av_next
        self.T1_stag = T1_stag
        self.p1_stag = p1_stag
        self.G = G
        self.n = n
        self.precision = precision
        self.geom = StageGeom()
        self.geom.h_rk_rel = h_rk_rel
        self.geom.h_na_rel = h_na_rel
        self.geom.delta_a_rk_rel = delta_a_rk_rel
        self.geom.delta_a_na_rel = delta_a_na_rel
        self.geom.const_diam_par = const_diam_par
        self.geom.d1_in_rel = d1_in_rel

    def compute(self):
        self._compute_gas_dynamics()
        self.geom.compute()

    def _compute_gas_dynamics(self):
        self.c_p_av = self.k_av * self.R_gas / (self.k_av - 1)
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
        self.geom.r1_av_rel = ((1 + self.geom.d1_in_rel**2) / 2) ** 0.5
        self.c1_u_rel = self.geom.r1_av_rel * (1 - self.R_av) - self.H_t_rel / (2 * self.geom.r1_av_rel)
        self.c1_u = self.c1_u_rel * self.u1_out
        self.c1 = np.sqrt(self.c1_u**2 + self.c1_a**2)
        self.alpha1 = np.arctan(self.c1_a_rel / self.c1_u_rel)
        self.lam1 = self.c1_a / (np.sin(self.alpha1) * self.a_cr1)
        self.q1 = gd.q(self.lam1, self.k_av)
        self.geom.F1 = self.G * (self.R_gas * self.T1_stag) ** 0.5 / (gd.m(self.k_av) * self.p1_stag * self.q1 *
                                                                      np.sin(self.alpha1))
        self.geom.D1_out = self.geom.get_D_out(self.geom.F1, self.geom.d1_in_rel)
        self.geom.D1_in = self.geom.d1_in_rel * self.geom.D1_out
        self.geom.D_const = self.geom.get_D_const(self.geom.d1_in_rel, self.geom.const_diam_par, self.geom.D1_out)
        self.alpha3 = self.alpha1
        self.u3_out = self.u1_out
        self.outlet_geom_res = 1
        while self.outlet_geom_res >= self.precision:
            self.alpha3_old = self.alpha3
            self.u3_out_old = self.u3_out
            self.alpha3, self.u3_out, self.outlet_geom_res = self._get_outlet_section_params(self.alpha3_old,
                                                                                             self.u3_out_old)
        self.c3_u = self.c3_u_rel * self.u3_out
        self.c3 = np.sqrt(self.c3_a + self.c3_u)
        self.geom.r2_av_rel = 0.5 * (self.geom.r1_av_rel + self.geom.r3_av_rel)
        self.c2_u_rel = (self.H_t_rel + self.c1_u_rel * self.geom.r1_av_rel) / self.geom.r2_av_rel
        self.c2_a = 0.5 * (self.c1_a + self.c3_a)
        self.u2_out = 0.5 * (self.u1_out + self.u3_out)
        self.c2_a_rel = self.c2_a / self.u2_out
        self.beta1 = np.arctan(self.c1_a_rel / (self.geom.r1_av_rel - self.c1_u_rel))
        self.beta2 = np.arctan(self.c2_a_rel / (self.geom.r2_av_rel - self.c2_u_rel))
        self.alpha2 = np.arctan(self.c2_a_rel / self.c2_u_rel)
        self.epsilon_rk = self.beta2 - self.beta1
        self.epsilon_na = self.alpha3 - self.alpha2
        self.w1 = self.c1_a / np.sin(self.beta1)
        self.w2 = self.c2_a / np.sin(self.beta2)
        self.w1_u = self.w1 * np.cos(self.beta1)
        self.w2_u = self.w2 * np.cos(self.beta2)
        self.c2 = self.c2_a / np.sin(self.alpha2)
        self.c2_u = self.c2_u_rel * self.u2_out
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
        self.c3_a = u3_out * self.c3_a_rel
        self.lam3 = self.c3_a / (np.sin(alpha3) * self.a_cr3)
        self.q3 = gd.q(self.lam3, self.k_av)
        self.geom.F3 = self.geom.F1 * (self.q1 * self.p1_stag * np.sin(self.alpha1)) / \
                       (self.q3 * self.p3_stag * np.sin(alpha3)) * np.sqrt(self.T3_stag / self.T1_stag)
        self.geom.d3_in_rel = self.geom.get_d_in_rel(self.geom.F3, self.geom.const_diam_par, self.geom.D_const)
        self.geom.r3_av_rel = ((1 + self.geom.d3_in_rel ** 2) / 2) ** 0.5
        self.c3_u_rel = self.geom.r3_av_rel * (1 - self.R_av_next) - self.H_t_rel_next / (2 * self.geom.r3_av_rel)
        if self.c3_u_rel != 0:
            alpha3_new = np.arctan(self.c3_a_rel / self.c3_u_rel)
        else:
            alpha3_new = np.arctan(np.inf)
        self.geom.D3_out = self.geom.get_D_out(self.geom.F3, self.geom.d3_in_rel)
        self.geom.D3_in = self.geom.D3_out * self.geom.d3_in_rel
        u3_out_new = np.pi * self.geom.D3_out * self.n / 60
        res = max(abs(alpha3 - alpha3_new) / alpha3, abs(u3_out - u3_out_new) / u3_out)
        return alpha3_new, u3_out_new, res

    def plot_velocity_triangle(self, fname: str=None, figsize=(6, 6)):
        plt.figure(figsize=figsize)
        plt.plot([0, self.c1_u, -self.w1_u, 0], [0, -self.c1_a, -self.c1_a, 0], label='inlet', lw=1.5, color='red')

        plt.plot([0, self.c2_u, -self.w2_u, 0], [0, -self.c2_a, -self.c2_a, 0], label='outlet', lw=1.5, color='blue')
        plt.grid()
        max_size = max(self.c1_a, self.c2_a, 2 * max(self.w1_u, self.c2_u))
        plt.xlim(-0.55 * max_size, 0.55 * max_size)
        plt.ylim(-max_size * 1.05, 0.05 * max_size)
        plt.legend(fontsize=12)
        plt.show()
        if fname:
            plt.savefig(fname)
