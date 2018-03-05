import unittest
from average_stream_line.stage_tools import Stage
from gas_turbine_cycle.gases import Air


class StageTests(unittest.TestCase):
    def setUp(self):
        self.stage = Stage(
            work_fluid=Air(),
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