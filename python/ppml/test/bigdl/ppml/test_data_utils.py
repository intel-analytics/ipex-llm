import unittest
import pandas as pd
from bigdl.ppml.data_utils import *


class MyTestCase(unittest.TestCase):
    def test_pandas_to_numpy(self):
        df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        array = convert_to_numpy(df)
        array


if __name__ == '__main__':
    unittest.main()
