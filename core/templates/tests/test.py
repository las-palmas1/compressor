from core.average_streamline.compressor import Compressor
from core.average_streamline.dist_tools import QuadraticBezier
from jinja2 import Template, FileSystemLoader, Environment, select_autoescape
import unittest
import core.templates
import core.templates.tests
import numpy as np
from gas_turbine_cycle.gases import Air
from scipy.interpolate import interp1d


class TestAverageLineTemplate(unittest.TestCase):
    def setUp(self):
        stage_num = 12
        self.H_t_rel_arr = QuadraticBezier(0.32, 0.33, stage_num, angle1=np.radians(10),
                                           angle2=np.radians(10)).get_array()
        self.eta_ad_stag_arr = QuadraticBezier(0.85, 0.86, stage_num, angle1=np.radians(10),
                                               angle2=np.radians(10)).get_array()
        self.c1_a_rel_arr = QuadraticBezier(0.45, 0.51, stage_num, angle1=np.radians(10),
                                            angle2=np.radians(-1)).get_array()
        self.h_rk_rel_dist = lambda z: interp1d([0, stage_num - 1], [2.5, 1.6])(z).__float__()
        self.h_na_rel_dist = lambda z: interp1d([0, stage_num - 1], [3.2, 1.8])(z).__float__()
        self.delta_a_rk_rel = lambda z: interp1d([0, stage_num - 1], [0.3, 0.5])(z).__float__()
        self.delta_a_na_rel = lambda z: interp1d([0, stage_num - 1], [0.3, 0.5])(z).__float__()

        self.compressor = Compressor(
            work_fluid=Air(),
            stage_num=stage_num,
            const_diam_par_arr=[0.5 for _ in range(stage_num)],
            p0_stag=1e5,
            T0_stag=288,
            G=40,
            n=11000,
            H_t_rel_arr=self.H_t_rel_arr,
            eta_ad_stag_arr=self.eta_ad_stag_arr,
            R_av_arr=QuadraticBezier.get_array_from_dist(lambda z: 0.5, stage_num),
            k_h_arr=QuadraticBezier.get_array_from_dist(lambda z: 0.98, stage_num),
            c1_a_rel_arr=self.c1_a_rel_arr,
            h_rk_rel_arr=QuadraticBezier.get_array_from_dist(self.h_rk_rel_dist, stage_num),
            h_na_rel_arr=QuadraticBezier.get_array_from_dist(self.h_na_rel_dist, stage_num),
            delta_a_rk_rel_arr=QuadraticBezier.get_array_from_dist(self.delta_a_rk_rel, stage_num),
            delta_a_na_rel_arr=QuadraticBezier.get_array_from_dist(self.delta_a_na_rel, stage_num),
            d1_in_rel1=0.5,
            zeta_inlet=0.04,
            zeta_outlet=0.35,
            c11_init=250,
        )
        self.compressor.compute()

    def test_template_rendering(self):
        loader = FileSystemLoader(
            [
                core.templates.__path__[0],
                core.templates.tests.__path__[0]
            ]
        )
        env = Environment(
            loader=loader,
            autoescape=select_autoescape(['tex']),
            block_start_string='</',
            block_end_string='/>',
            variable_start_string='<<',
            variable_end_string='>>',
            comment_start_string='<#',
            comment_end_string='#>'
        )

        template = env.get_template('test_report_templ.tex')
        content = template.render(
            comp=self.compressor
        )
        with open('test_report.tex', 'w', encoding='utf-8') as f:
            f.write(content)