# Chronos Known Issue
## **1. Issue 1**
**Problem description**

Numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject. 

**Solution**
* `pip install -y pycocotools`
* `pip install pycocotools --no-cache-dir --no-binary :all:`
* `conda install –c conda-forge pycocotools`

---------------------------

## **2. Issue 2**
**Problem description**

NotImplementedError: Cannot convert a symbolic Tensor (encoder_lstm_8/strided_slice:0) to a numpy array. 

**Solution**

* `pip install numpy==1.19.5`

---------------------------

## **3. Issue 3**

**Problem description**

StanModel object has no attribute 'fit_class', cause of pip, may be. 

**Solution**

* `pip uninstall pystan prophet –y`
* `conda install –c conda-forge prophet=1.0.1`

---------------------------

## **4. Issue 4**
**Problem description**

Exception: No active RayContext. Please call init_orca_context to create a RayContext.
> ray_ctx = RayContext.get()<br>
> ray_ctx = RayContext.get(initialize=False)

**Solution**

* Make sure all operations are before `stop_orca_context`. 
* No other `RayContext` exists before `init_orca_context`. 

---------------------------

## **5. Issue 5**
**Problem description**

 Sed: error while loading shared libraries: libunwind.so.8: cannot open shared object file: No such file or directory.
> Only happens when you run `source bigdl-nano-init`. 

**Solution**

* `apt-get install libunwind8-dev` 

---------------------------
