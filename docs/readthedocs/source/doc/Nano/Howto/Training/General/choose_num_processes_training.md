# Choose the Number of Porcesses for Multi-Instance Training

BigDL-Nano supports multi-instance training on a server with multiple CPU cores or sockets. With Nano, you could launch a self-defined number of processes to perform data-parallel training. When choosing the number of processes, there are 3 empirical recommendations for better training performance:

1. There should be at least 7 CPU cores assigned to each process.
2. For multiple sockets, the CPU cores assiged to each process should belong to the same socket (due to NUMA issue). That is, the number of CPU cores per process should be a divisor of the number of CPU cores placed in each sockets.
3. Only physical CPU cores should be considered (do not count in CPU cores for hyperthreading).

```eval_rst
.. note:: 
    By default, Nano will distribute CPU cores evenly among processes.
```

Here is an example. Suppose we have a sever with 2 sockets. Each socket has 28 physical CPU cores. For this case, the number of CPU cores per process c should satisfiy:

```eval_rst
.. math::
    \begin{cases}
    c \text{ is divisor of } 28 \\
    c \ge 7 \\
    \end{cases} \Rightarrow 
    c \in \{7, 14, 28\}
``` 

Based on that, the number of processes np can be calculated as:

```eval_rst
.. math::
    \begin{cases}
    np = \frac{28+28}{c}\ , c \in \{7, 14, 28\} \\
    np > 1 \\
    \end{cases} \Rightarrow np = \text{8 or 4 or 2}
``` 

That is, empirically, we could set the number of processes to 2, 4 or 8 here for good training performance.

```eval_rst
.. card::

    **Related Readings**
    ^^^
    * `How to accelerate a PyTorch Lightning application on training workloads through multiple instances <../PyTorchLightning/accelerate_pytorch_lightning_training_multi_instance.html>`_
    * `How to accelerate a TensorFlow Keras application on training workloads through multiple instances <../TensorFlow/accelerate_tensorflow_training_multi_instance.html>`_
```