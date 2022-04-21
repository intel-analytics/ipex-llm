# PPML FGBoost Benchmark test

Start the FLServer
```
from bigdl.ppml import *
fl_server = FLServer()
fl_server.build()
fl_server.start()
```

Start the FGBoost Regression with arguments
* data_size: the size of dummy data, default 100
* data_dim: the dimension of dummy data, default 10
* num_round: the number of boosting round, default 10

e.g. start the benchmark test with data size 10000, data dimension 100, boosting round 5
```
python fgboost_benchmark.py --data_size 10000 --data_dim 100 --num_round 5
```
