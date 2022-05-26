# FGBoost Regression Python Example

### Walkthrough
We provide the full code below, which you could directly run once BigDL PPML [installation](#link) done.
```python
import numpy as np
from bigdl.ppml import FLServer
from bigdl.ppml.algorithms.fgboost_regression import FGBoostRegression
from bigdl.ppml.utils import init_fl_context

fl_server = FLServer()
fl_server.build()
fl_server.start()
init_fl_context()

x, y = np.ones([2, 3]), np.ones([2])
fgboost_regression = FGBoostRegression()
fgboost_regression.fit(x, y)
result = fgboost_regression.predict(x)
result

fl_server.close()
```
Now we dive into the code.
### Start FLServer
To start a BigDL PPML application, you first start a FLServer by
```python
fl_server = FLServer()
fl_server.build()
fl_server.start()
```
### Initialize FLContext
The client to interact with FLServer is inside FLContext, to use it, initialize the FLContext by
```python
init_fl_context()
```
### Run Algorithm
Then create a `FGBoostRegression` instance to apply Federated Gradient Boosting Regression algorithm, and call train and predict on dummy data.
```python
fgboost_regression = FGBoostRegression()
fgboost_regression.fit(x, y)
result = fgboost_regression.predict(x)
result
```

### Model Save and Load
fgboost_regression.save_model(dest_file_path)
fgboost_regression.load_model(src_file_path)
