# PyTorch CUDA Patch

BigDL-Nano also provides CUDA patch (`bigdl.nano.pytorch.patching.patch_cuda`) to help you run CUDA code without GPU. This patch will replace CUDA operations with equivalent CPU operations, so after applying it, you can run CUDA code on your CPU without changing any code.

```eval_rst
.. tip::
    There is also ``bigdl.nano.pytorch.patching.unpatch_cuda`` to unpatch it.
```

You can use it as following:
```python
from bigdl.nano.pytorch.patching import patch_cuda, unpatch_cuda
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

