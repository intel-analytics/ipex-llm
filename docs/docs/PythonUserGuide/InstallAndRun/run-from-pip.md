## **Precondition**

* [Install analytics-zoo from pip](install-from-pip.md)
* Only __Python 2.7__, __Python 3.5__ and __Python 3.6__ are supported for now.

---
## **Use an Interactive Shell**
* Type `python` in the command line to start a REPL.
* Try to run the [example code](#example-code) to verify the installation.

---
## **Use Jupyter Notebook**
* Just start jupyter notebook as you normally do, e.g.

```bash
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

* Try to run the [example code](#example-code) to verify the installation.

---
## **Example code**

To verify if Analytics Zoo can run successfully, run the following simple code:

```python
import zoo.version
from zoo.common.nncontext import *
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.keras.layers import *

# Get the current Analytics Zoo version
zoo.version.__version__
# Create or get a SparkContext. This will also init the BigDL engine.
sc = get_nncontext()
# Create a Sequential model containing a Dense layer.
model = Sequential()
model.add(Dense(8, input_shape=(10, )))
```

## **Configurations**

* Increase memory

```bash
export SPARK_DRIVER_MEMORY=20g
```

* Add extra jars or python packages

 &emsp; Set the environment variables `BIGDL_JARS` and `BIGDL_PACKAGES` __BEFORE__ creating `SparkContext`:
```bash
export BIGDL_JARS=...
export BIGDL_PACKAGES=...
```