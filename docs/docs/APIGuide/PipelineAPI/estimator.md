## Estimator


`Estimator` supports the training and evaluation of BigDL models, Keras-like models and PyTorch models. It wraps a model, and provide a uniform training, evaluation, or prediction operation on both localhost and distributed spark environment.


### Creating an Estimator

In summary, you need to supply three parameters to create an Estimator: 1)a model, 2) optimMethod
(s), 3) model directory, as shown below:

**Scala:**

```scala
val estimator = Estimator[T: ClassTag](
      model: Module[T], 
      optimMethods: Map[String, OptimMethod[T]] = Map(), 
      modelDir: Option[String] = None)
```

`T`: the numeric type(Float/Double).  
`model`: the model will be optimized.  
`optimMethods`: the methods to optimize the model. Submodule names and optimMethod pairs.  
`modelDir`(Optional): model checkpoint directory, and related summary directory.

```scala
val estimator = Estimator[T: ClassTag](
      model: Module[T],
      optimMethod: OptimMethod[T],
      modelDir: Option[String] = None)

```

`T`: the numeric type(Float/Double).  
`model`: the model will be optimized.  
`optimMethod`: the method to optimize the model.  
`modelDir`(Optional): model checkpoint directory, and related summary directory.

**Python:**

```python
estimator = Estimator(model, optim_methods, model_dir)

```

`model`: the model will be optimized.  
`optim_methods`: the methods to optimize the model. Both single optimMethod and Dict(submodule 
name, optimMethod) are supported.  
`model_dir`(Optional): model checkpoint directory, and related summary directory.

### Training
Train the model with provided trainSet and criterion. The training will end when the endTrigger is 
triggered. During the training, if the checkPointTrigger is defined and triggered, the model will be saved to modelDir. And if validationSet and validationMethod are defined, the model will be evaluated at the checkpoint.

**Scala:**
```scala
estimator.train(trainSet: FeatureSet[MiniBatch[T]],
            criterion: Criterion[T],
            endTrigger: Option[Trigger] = None,
            checkPointTrigger: Option[Trigger] = None,
            validationSet: FeatureSet[MiniBatch[T]] = null,
            validationMethod: Array[ValidationMethod[T]] = null)
```

`trainSet`: training dataset in type of FeatureSet[MiniBatch].  
`criterion`: loss function.  
`endTrigger`: when to finish the training.  
`checkPointTrigger`: how often to save a checkpoint and evaluate the model.  
`validationSet`: validation dataset in type of FeatureSet[MiniBatch].  
`validationMethod`: a set of validationMethod.

**Python:**
##### Train Samples
```python
estimator.train(train_set, criterion, end_trigger, checkpoint_trigger,
              validation_set, validation_method, batch_size)
```
`train_set`: training dataset in type of FeatureSet[Sample[T]].  
`criterion`: loss function.  
`end_trigger`: when to finish the training.  
`checkpoint_trigger`: how often to save a checkpoint and evaluate the model.  
`validation_set`: validation dataset in type of FeatureSet[Sample[T]].  
`validation_method`: a set of validationMethod.  
`batch_size`: mini batch size.


##### Train ImageFeatures
```python
estimator.train_imagefeature(self, train_set, criterion, end_trigger, checkpoint_trigger,
                           validation_set, validation_method, batch_size)
```
`train_set`: training dataset in type of FeatureSet[ImageFeature].  
`criterion`: loss function.  
`end_trigger`: when to finish the training.  
`checkpoint_trigger`: how often to save a checkpoint and evaluate the model.  
`validation_set`: validation dataset in type of FeatureSet[ImageFeature].  
`validation_method`: a set of validationMethod.  
`batch_size`: mini batch size.


### Evaluation
Evaluate the model on the validationSet with the validationMethods.

**Scala:**
```scala
estimator.evaluate(validationSet: FeatureSet[MiniBatch[T]],
                   validationMethod: Array[ValidationMethod[T]])
```
`validationSet`: validation dataset in type of FeatureSet.  
`validationMethod`: a set of validationMethod.

**Python:**
##### Evaluate Samples
```python
estimator.evaluate(validation_set, validation_method, batch_size)
```
`validation_set`: validation dataset in type of FeatureSet[Sample[T]].  
`validation_method`: a set of validationMethod.  
`batch_size`: mini batch size.

##### Train ImageFeatures
```python
estimator.evaluate_imagefeature(validation_set, validation_method, batch_size)
```
`validation_set`: validation dataset in type of FeatureSet[ImageFeature].  
`validation_method`: a set of validationMethod.  
`batch_size`: mini batch size.

### Other Important API

#### setConstantGradientClipping
Set constant gradient clipping during the training process. In order to take effect, it needs to 
be called before fit.

**Scala:**
```scala
estimator.setConstantGradientClipping(min: Double, max: Double)
```
`min`: The minimum value to clip by. Double.  
`max`: The maximum value to clip by. Double.

**Python:**
```python
estimator.set_constant_gradient_clipping(min, max)
```
`min`: The minimum value to clip by. Double.  
`max`: The maximum value to clip by. Double.

#### setGradientClippingByL2Norm
Set clip gradient to a maximum L2-Norm during the training process. In order to take effect, it 
needs to be called before fit.

**Scala:**
```scala
estimator.setGradientClippingByL2Norm(clipNorm: Double)
```
`clipNorm`: Gradient L2-Norm threshold. Double.

**Python:**
```python
estimator.set_l2_norm_gradient_clipping(clip_norm)
```
`clip_norm`: Gradient L2-Norm threshold. Double.

#### clearGradientClipping 
Clear gradient clipping parameters. In this case, gradient clipping will not be applied. In order
 to take effect, it needs to be called before fit.
 
 **Scala:**
 ```scala
 estimator.clearGradientClipping()
 ```
 
 **Python:**
 ```python
estimator.clear_gradient_clipping()
```

### Examples
Please refer to [Inception(Scala)](https://github.com/intel-analytics/analytics-zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/inception/Train.scala) or [Inception(Python)](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/examples/inception/inception.py) for 
more information
