# CUDA Patch

BigDL-Nano also provides CUDA patch to help you run CUDA code without GPU. This patch will replace CUDA operations with equivalent CPU operations, so after applying it, you can run CUDA code on your CPU without changing any code.

This patch is included in `bigdl.nano.pytorch.patch_torch`, you can call `patch_torch(cuda_to_cpu=True)` to enable it (the `cuda_to_cpu` parameter defaults to `True`). You can also use it alone by `bigdl.nano.pytorch.patching.gpu_cpu.patch_cuda`.

```eval_rst
.. tip::
    There are also ``bigdl.nano.pytorch.unpatch_torch`` and ``bigdl.nano.pytorch.patching.gpu_cpu.unpatch_cuda`` to unpatch it.
```

You can use it as following:
```eval_rst

.. tabs::

    .. tab:: patch_torch

        .. code-block:: python

            from bigdl.nano.pytorch import patch_torch, unpatch_torch
            patch_torch()   # The ``cuda_to_cpu`` parameter defaults to ``True``

            # Then you can run CUDA code directly even without GPU
            model = torchvision.models.resnet50(pretrained=True).cuda()
            inputs = torch.rand((1, 3, 128, 128)).cuda()
            with torch.no_grad():
                outputs = model(inputs)

            unpatch_torch()


    .. tab:: patch_cuda

        .. code-block:: python

            from bigdl.nano.pytorch.patching.gpu_cpu import patch_cuda, unpatch_cuda
            patch_cuda()

            # Then you can run CUDA code directly even without GPU
            model = torchvision.models.resnet50(pretrained=True).cuda()
            inputs = torch.rand((1, 3, 128, 128)).cuda()
            with torch.no_grad():
                outputs = model(inputs)

            unpatch_cuda()


```

```eval_rst
.. note::
    - You should apply this patch at the beginning of your code, because it can only affect the code after calling it.
    - This CUDA patch is incompatible with JIT, applying it will disable JIT automatically.
```

