Analytics-Zoo provides a GANEstimator to support training GAN like models.
Currently we support standard unconditional/conditional GAN and other GAN types will be supported in the future.


---
## **GANEstimator**

**Python**
```python
GANEstimator(generator_fn,
             discriminator_fn,
             generator_loss_fn,
             discriminator_loss_fn,
             generator_optimizer,
             discriminator_optimizer,
             generator_steps=1,
             discriminator_steps=1,
             model_dir=None,
    )
```

**Arguments**

* **generator_fn**: a python function that defines the generator. It should takes a single noise tensor (unconditional)
a tuple of tensors in which the first element represents noise and the second label (conditional) and return the
generated data.  
* **discriminator_fn**: a python function that defines the discriminator. The discriminator_fn should have two inputs.
The first input should be the real data or generated data. The inputs to generator will also be passed too discriminator
as the second input.
* **generator_loss_fn**: the loss function on the generator. It should take the output of discriminator on generated data
and return the loss for generator.
* **discriminator_loss_fn**: the loss function on the discriminator. The discriminator_loss_fn should have two inputs. The
first input is the output of discriminator on generated data and the second input is the output of discriminator on real data.
* **generator_optimizer**: the optimizer to optimize generator, should be an instance of tf.train.Optimizer
* **discriminator_optimizer**: the optimizer to optimizer discriminator, should be an instance of tf.train.Optimizer
* **generator_steps**: the number of consecutive steps to run generator in each round
* **discriminator_steps**: the number of consecutive steps to run discriminator in each round


### train

```python
estimator.train(input_fn=input_fn, end_trigger=MaxIteration(100))
```

**Arguments**

* **input_fn**: a python function that takes zero arguments and return a TFDataset. Each record in the TFDataset should
a tuple. The first element of the tuple is generator inputs, and the second element of the tuple should be real data.
* **end_trigger**: BigDL's [Trigger](https://bigdl-project.github.io/0.9.0/#APIGuide/Triggers/) to indicate when to stop the training. If none, defaults to
train for one epoch.



