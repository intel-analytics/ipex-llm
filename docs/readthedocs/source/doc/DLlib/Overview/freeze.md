## Model Freeze
To "freeze" a model means to exclude some layers of model from training.

```scala
model.freeze("layer1", "layer2")
model.unFreeze("layer1", "layer2")
```
* The model can be "freezed" by calling ```freeze()```. If a model is freezed,
its parameters(weight/bias, if exists) are not changed in training process.
If model names are passed, then layers that match the given names will be freezed.
* The whole model can be "unFreezed" by calling ```unFreeze()```.
If model names are provided, then layers that match the given names will be unFreezed.
* stop the input gradient of layers that match the given names. Their input gradient are not computed.
And they will not contributed to the input gradient computation of layers that depend on them.

Original model without "freeze"
```scala
val reshape = Reshape(Array(4)).inputs()
val fc1 = Linear(4, 2).setName("fc1").inputs()
val fc2 = Linear(4, 2).setName("fc2").inputs(reshape)
val cadd_1 = CAddTable().setName("cadd").inputs(fc1, fc2)
val output1_1 = ReLU().inputs(cadd_1)
val output2_1 = Threshold(10.0).inputs(cadd_1)

val model = Graph(Array(reshape, fc1), Array(output1_1, output2_1))

val input = T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
  Tensor(T(0.5f, 0.4f, -0.2f, -0.1f)))
val gradOutput = T(Tensor(T(1.0f, 2.0f)), Tensor(T(3.0f, 4.0f)))

fc1.element.getParameters()._1.apply1(_ => 1.0f)
fc2.element.getParameters()._1.apply1(_ => 2.0f)
model.zeroGradParameters()
println("output1: \n", model.forward(input))
model.backward(input, gradOutput)
model.updateParameters(1)
println("fc2 weight \n", fc2.element.parameters()._1(0))
```
```
(output1:
, {
	2: 0.0
	   0.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	1: 2.8
	   2.8
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
 })
(fc2 weight
,1.9	1.8	2.3	2.4
1.8	1.6	2.6	2.8
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4])
```

"Freeze" ```fc2```, the parameters of ```fc2``` is not changed.
```scala
fc1.element.getParameters()._1.apply1(_ => 1.0f)
fc2.element.getParameters()._1.apply1(_ => 2.0f)
model.zeroGradParameters()
model.freeze("fc2")
println("output2: \n", model.forward(input))
model.backward(input, gradOutput)
model.updateParameters(1)
println("fc2 weight \n", fc2.element.parameters()._1(0))
```

```
(output2:
, {
	2: 0.0
	   0.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	1: 2.8
	   2.8
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
 })
(fc2 weight
,2.0	2.0	2.0	2.0
2.0	2.0	2.0	2.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4])
```
"unFreeze" ```fc2```, the parameters of ```fc2``` will be updated.
```scala
fc1.element.getParameters()._1.apply1(_ => 1.0f)
fc2.element.getParameters()._1.apply1(_ => 2.0f)
model.zeroGradParameters()
model.unFreeze()
println("output3: \n", model.forward(input))
model.backward(input, gradOutput)
model.updateParameters(1)
println("fc2 weight \n", fc2.element.parameters()._1(0))
```
```
(output3:
, {
	2: 0.0
	   0.0
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
	1: 2.8
	   2.8
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 2]
 })
(fc2 weight
,1.9	1.8	2.3	2.4
1.8	1.6	2.6	2.8
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x4])
```
