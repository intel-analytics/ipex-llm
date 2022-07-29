## ConstantGradientClipping ##

Set constant gradient clipping during the training process.

```scala
model.setConstantGradientClipping(clipNorm)
```
param:
   * min: The minimum value to clip by.
   * max: The maximum value to clip by.

## GradientClippingByL2Norm ##

Clip gradient to a maximum L2-Norm during the training process.

```scala
model.setGradientClippingByL2Norm(clipNorm)
```
param:
   * clipNorm: Gradient L2-Norm threshold
