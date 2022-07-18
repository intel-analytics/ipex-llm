#!/usr/bin/env python3

from graminelibos.regression import RegressionTestCase

class TC_00_Entrypoint(RegressionTestCase):
    def test_010_atexit_func(self):
        self.run_binary(['atexit_func'])

    def test_020_fpu_control_word(self):
        self.run_binary(['fpu_control_word'])

    def test_030_rflags(self):
        self.run_binary(['rflags'])

    def test_040_mxcsr(self):
        self.run_binary(['mxcsr'])

    def test_050_stack(self):
        self.run_binary(['stack'])

    def test_060_arg(self):
        self.run_binary(['stack_arg', 'foo', 'bar'])

    def test_070_env(self):
        self.run_binary(['stack_env'])

    def test_080_auxv(self):
        self.run_binary(['stack_auxiliary'])
