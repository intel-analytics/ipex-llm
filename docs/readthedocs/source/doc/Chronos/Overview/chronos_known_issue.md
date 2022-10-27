# Chronos Known Issue

## Version Compatibility Issues

### Numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject

**Problem description**

It seems to be a numpy compatibility issue, we do not recommend to solve it by downgrading Numpy to 1.19.x,
when no other issues exist, the solution is given below.

**Solution**
* `pip install -y pycocotools`
* `pip install pycocotools --no-cache-dir --no-binary :all:`
* `conda install –c conda-forge pycocotools`

---------------------------

### Cannot convert a symbolic Tensor (encoder_lstm_8/strided_slice:0) to a numpy array

**Problem description**

This is a compatibility issue caused by Tensorflow and Numpy 1.20.x

**Solution**

* `pip install numpy==1.19.5`

---------------------------

### StanModel object has no attribute 'fit_class'

**Problem description**

We recommend reinstalling prophet using conda or miniconda.

**Solution**

* `pip uninstall pystan prophet –y`
* `conda install –c conda-forge prophet=1.0.1`

---------------------------

## Dependency Issues

### RuntimeError: No active RayContext

**Problem description**

Exception: No active RayContext. Please call init_orca_context to create a RayContext.
> ray_ctx = RayContext.get()<br>
> ray_ctx = RayContext.get(initialize=False)

**Solution**

* Make sure all operations are before `stop_orca_context`. 
* No other `RayContext` exists before `init_orca_context`. 

---------------------------

### error while loading shared libraries: libunwind.so.8: cannot open shared object file: No such file or directory.

**Problem description**

A dependency is missing from your environment, only happens when you run `source bigdl-nano-init`.

**Solution**

* `apt-get install libunwind8-dev` 

---------------------------
