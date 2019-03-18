# How to Use Lower Numerical Precision Deep Learning Inference in BigDL


You can use the mkldnn version of low numerical precision inference by the
API `quantize()` which will give you better performance on Intel Xeon Scalable
processors. There're only two steps, scale generation and quantize the model.

## Generate the Scales of Pretrained Model

If you use a BigDL model which is trained by yourself or converted from other
frameworks. You should generate the scales first. It needs some sample images
to do the `forward` which can be the test or validation dataset.

After that, you can call `GenerateInt8Scales`, it will generate a model with
a `quantized` as the suffix. It's the original model with scales information.

```bash
```

## Do the Evaluation on the Quantized Model

When you prepared the relative quantized model, it's very simple to use int8 based
on mkldnn. On your loaded model, to call `quantize()`, it will return a new
quantized model. Now, you can do inference like other models. You should enable the
model fusion by java property, `-Dbigdl.mkldnn.fusion=true`, which will have the
best performance.

```bash
```