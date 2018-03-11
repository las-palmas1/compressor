import unittest
from average_stream_line.stage_tools import Stage
from gas_turbine_cycle.gases import Air
from average_stream_line.compressor import Compressor
from average_stream_line.dist_tools import QuadraticBezier
import numpy as np


class StageTests(unittest.TestCase):
    def setUp(self):
        self.stage = Stage(
            k_av=1.4,
            R_gas=287,
            H_t_rel=0.34,
            H_t_rel_next=0.33,
            u1_out=300,
            k_h=0.95,
            eta_ad_stag=0.89,
            d1_in_rel=0.45,
            c1_a_rel=0.5,
            c3_a_rel=0.45,
            R_av=0.5,
            R_av_next=0.5,
            T1_stag=320,
            p1_stag=1e5,
            G=45,
            n=11000,
            h_rk_rel=2.5,
            h_na_rel=3.0,
            delta_a_rk_rel=0.3,
            delta_a_na_rel=0.3,
            const_diam_par=0.5
        )
        self.stage.compute()

    def test_velocity_triangle(self):
        self.stage.plot_velocity_triangle()

    def test_geom_computing(self):
        self.assertNotEqual(None, self.stage.geom.D1_out)
        self.assertNotEqual(None, self.stage.geom.D1_av)
        self.assertNotEqual(None, self.stage.geom.D1_in)
        self.assertNotEqual(None, self.stage.geom.F1)
        self.assertNotEqual(None, self.stage.geom.const_diam_par)
        self.assertNotEqual(None, self.stage.geom.d1_in_rel)
        self.assertNotEqual(None, self.stage.geom.D15_out)
        self.assertNotEqual(None, self.stage.geom.D15_in)
        self.assertNotEqual(None, self.stage.geom.d15_in_rel)
        self.assertNotEqual(None, self.stage.geom.F15)
        self.assertNotEqual(None, self.stage.geom.D20_out)
        self.assertNotEqual(None, self.stage.geom.D20_in)
        self.assertNotEqual(None, self.stage.geom.d20_in_rel)
        self.assertNotEqual(None, self.stage.geom.F20)
        self.assertNotEqual(None, self.stage.geom.D25_out)
        self.assertNotEqual(None, self.stage.geom.D25_in)
        self.assertNotEqual(None, self.stage.geom.d25_in_rel)
        self.assertNotEqual(None, self.stage.geom.r1_av_rel)
        self.assertNotEqual(None, self.stage.geom.r2_av_rel)
        self.assertNotEqual(None, self.stage.geom.r3_av_rel)
        self.assertNotEqual(None, self.stage.geom.h_na)
        self.assertNotEqual(None, self.stage.geom.h_rk)
        self.assertNotEqual(None, self.stage.geom.b_a_na)
        self.assertNotEqual(None, self.stage.geom.b_a_rk)
        self.assertNotEqual(None, self.stage.geom.delta_a_na)
        self.assertNotEqual(None, self.stage.geom.delta_a_rk)
        self.assertNotEqual(None, self.stage.geom.D_const)
        self.assertNotEqual(None, self.stage.geom.length)


class CompressorTests(unittest.TestCase):
    def setUp(self):
        stage_num = 12
        self.H_t_rel_dist = QuadraticBezier(0.32, np.radians(10), 0.33, np.radians(10), stage_num).get_dist()
        self.eta_ad_stag_dist = QuadraticBezier(0.85, np.radians(10), 0.86, np.radians(10), stage_num).get_dist()
        self.c1_a_rel_dist = QuadraticBezier(0.45, np.radians(10), 0.51, np.radians(-1), stage_num).get_dist()
        self.compressor = Compressor(
            work_fluid=Air(),
            stage_num=stage_num,
            const_diam_par=0.5,
            p0_stag=1e5,
            T0_stag=288,
            G=40,
            n=11000,
            H_t_rel_dist=self.H_t_rel_dist,
            eta_ad_stag_dist=self.eta_ad_stag_dist,
            R_av_dist=lambda z: 0.5,
            k_h_dist=lambda z: 0.98,
            c1_a_rel_dist=self.c1_a_rel_dist,
            d1_in_rel1=0.5,
            zeta_inlet=0.04,
            zeta_outlet=0.35,
            c11_init=250,
        )
        self.compressor.compute()

    def test_init_param_dist(self):
        self.compressor.plot_init_param_dist('H_t_rel')
        self.compressor.plot_init_param_dist('c1_a_rel')
        self.compressor.plot_init_param_dist('eta_ad_stag')

    def test_geometry_plot(self):
        self.compressor.plot_geometry()

    def test_temp_and_press_dist(self):
        self.compressor.plot_temp_dist()
        self.compressor.plot_press_dist()

    def test_stage_parameters_transfer(self):
        for i in range(len(self.compressor) - 1):
            stage = self.compressor[i]
            next_stage = self.compressor[i + 1]
            self.assertEqual(next_stage.u1_out, stage.u3_out)
            self.assertEqual(next_stage.geom.d1_in_rel, stage.geom.d3_in_rel)
            self.assertEqual(next_stage.T1_stag, stage.T3_stag)
            self.assertEqual(next_stage.p1_stag, stage.p3_stag)
            self.assertLess((next_stage.c1_a - stage.c3_a) / stage.c3_a, self.compressor.precision)
            self.assertEqual(next_stage.a_cr1, stage.a_cr3)
            self.assertEqual(next_stage.geom.r1_av_rel, stage.geom.r3_av_rel)
            self.assertEqual(next_stage.R_av, stage.R_av_next)
            self.assertEqual(next_stage.H_t_rel, stage.H_t_rel_next)
            self.assertEqual(next_stage.c1_u_rel, stage.c3_u_rel)
            self.assertEqual(next_stage.c1_u, stage.c3_u)
            self.assertEqual(next_stage.alpha1, stage.alpha3)
            self.assertLess((next_stage.lam1 - stage.lam3) / stage.lam3, self.compressor.precision)
            self.assertLess((next_stage.geom.F1 - stage.geom.F3) / stage.geom.F3, self.compressor.precision)
            self.assertLess((next_stage.geom.D_const - stage.geom.D_const) / stage.geom.D_const,
                            self.compressor.precision)
            self.assertLess((next_stage.geom.D1_out - stage.geom.D3_out) / stage.geom.D3_out, self.compressor.precision)
            self.assertLess((next_stage.geom.D1_in - stage.geom.D3_in) / stage.geom.D3_in, self.compressor.precision)




