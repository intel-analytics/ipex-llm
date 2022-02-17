import unittest
from bigdl.ppml import FLServer


class MyTestCase(unittest.TestCase):
    def test_fl_server_default_config(self):
        fl_server = FLServer()

    def test_fl_server_custom_config(self):
        pass


if __name__ == '__main__':
    unittest.main()
