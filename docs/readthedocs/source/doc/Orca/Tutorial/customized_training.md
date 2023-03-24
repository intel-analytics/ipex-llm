# How to Customize Your Pytorch Training Process

This tutorial provides a guide on customizing your PyTorch training process with the callbacks mechanism.


## Callback Mechanism Introduction

### What is callback mechanism?

The callback mechanism involves passing a function as an argument into another function, which is then executed within the outer function to accomplish a specific task or action.

### How does a callback work?

The callback mechanism breaks down the entire training process into several stages, allowing users to customize different training processes by overriding the corresponding callbacks.

### How many callbacks are there?

There are two types of callbacks:

The first type is a callback that does nothing by default, such as `on_run_end`. We can use our existing Callback class to implement this.
```python
class Callback:
    
    def on_run_begin(self, runner): ...
    def on_run_end(self, runner): ...
    def on_epoch_begin(self, runner): ...
    def on_epoch_end(self, runner): ...
    def on_iter_begin(self, runner): ...
    def on_iter_end(self, runner): ...
```

Another is the callback that contains default behaviors defined by orca: on_iter_forward, on_iter_backward and on_lr_adjust. These methods are somewhat special, because only one MainCallback should be allowed to implement these methods among all callbacks(otherwise there will propagate forward and backward twice).
```python
class MainCallback(Callback);

    def on_iter_forward(self, runner):
        inputs, targets = runner.batch
        runner.outputs = runner.model(inputs)
        inputs, targets = runner.batch
        runner.loss = runner.criterion(runner.outputs, targets)

    def on_iter_backward(self, runner):
        runner.optimizer.zero_grad()
        runner.loss.backward()
        runner.optimizer.step()
    
    def on_lr_adjust(self, runner):
        if runner.lr_scheduler is not None:
           runner.lr_scheduler.step()

    def on_train_forward(self, runner): # By default it will call on_iter_forward.
        self.on_iter_forward(runner)

    def on_val_forward(self, runner): # By default it will call on_iter_forward.
        self.on_iter_forward(runner)
```

In general, there are 24 points where callbacks can be inserted from the beginning to the end of model training:

* global points: `before_run`, `after_run`, `before_epoch`, `after_epoch`, `before_iter`, `after_iter`, *`on_iter_forward`
* training points: `before_train_epoch`, `after_train_epoch`,  `before_train_iter`, `before_val_iter`, *`on_train_forward`, *`on_iter_backward`, *`on_lr_adjust`
* validation points: `before_val_epoch`, `after_val_epoch`, `before_val_iter`, `after_val_iter`, *`on_val_forward`
* prediction points: `before_pred_epoch`, `after_pred_epoch`, `before_pred_iter`, `after_pred_iter`, `on_pred_forward`

Note that points marked with star are only available in `MainCallBack`.

### How are these callbacks composed?

Basically we assume the **training** process as the following pseudocode:
```python

# In TorchRunner.train_epochs:

...

call_hooks("on_run_begin", self)

for epoch in range(epochs):

    self.epoch+=1
    call_hooks("on_epoch_begin", self)

    for batch_idx, batch in enumerate(self.dataloader):
        self.batch = batch
        self.batch_idx = batch_idx
        call_hooks("on_iter_begin", self)

        call_hooks("on_iter_forward", self)

        call_hooks("on_iter_backward", self)  # this will be called when training

        call_hooks("on_iter_end", self)

        # We should not present items in last iteration to users in the next iteration.
        if hasattrs(self, 'batch'|'batch_idx'|'outputs'|'loss')
           del self.batch
           del self.batch_idx
           del self.outputs
           del self.loss

    call_hooks("on_lr_adjust", self)  # this will be called when training

    call_hooks("on_epoch_end", self)

call_hooks("on_run_end", self)

```

### What attributes user can access during the whole process?

There are attributes that you can access **all the time** and others that can only be accessed **in some stages**:

Globally Access:
* num_epochs: Int. Total epochs to be trained.
* epochs: Int. Current epoch number.
* train_loader: Torch dataloader. The train dataloader returned by train_dataloader_creator.
* val_loader: Torch dataloader. The validation dataloader returned by val_dataloader_creator.
* rank: Int. The rank of this runner.
* model: Torch module. The model returned by model_creator.
* optimizer: Torch optimaizer. The optimizer returned by optimizer_creator.
* scheduler: Torch scheduler. The scheduler returned by scheduler_creator.
* criterion: The loss function returned by loss_creator.
* stop: Boolean. Whether to stop training in this iteration.
* global_step: Int. The total number of iterations.

Can be accessed within iterations(like `before_train_iter`, `after_train_iter` etc.):
* batch: The data batch of this iteration.
* batch_idx: The batch index of this iteration.
* output: The output of the model in this iteration.
* loss: The loss calculated in this iteration.

### How can users save and load their customized attributes?

Our Torch Runner provides `put&get` APIs to save and load user's customized attributes. Users can utilize such a storage place to assign and fetch their own attributes across hooks.

* runner.put(k, v): store a new value `v` with the key `k`.
* runner.update(k, v): update the value under key `k` with a new value `v`.
* runner.get(k): return the value under key `k`.
* runner.remove(k): remove the value under key `k`.

```python
#  Usage: Forward with dummy data.
class ResNetPerfCallback(MainCallback):
    def before_val_epoch(self, runner):
        ...
        if runner.config["dummy"]:
            runner.put("images", images)
            runner.put("target", target)
        ...
    
    def on_val_forward(self, runner):
        if runner.config["dummy"]:
            images = runner.get("images")
            target = runner.get("target")
        ...
        output, target, loss = self.forward(runner, images, target, warmup=False)
```


## Callback Usage Examples

You only need to pass callbacks to `est.fit`, `est.evaluate` and `est.predict` as parameters to let the runner do the work.
```python
est = Estimator.from_torch(...)

# Will automatically detect whether there is a unique mainHook, and put it in the first place in runner._hook if so.
est.fit(...,
        callbacks=[CustomMainCB(), Hook_1(), Hook_2(), ...])

# if you don't need to modify training process

est.evaluate(...,
        callbacks=[hook_1, hook_2, ...])

```

### Usage 1
Some popular image models like Mask-RCNN for object detection calculate the loss in a slightly different way that they calculate the loss inside the model.
```python
loss_dict = model(images, targets)
losses = sum(loss for loss in loss_dict.values())
```

We can implement the following logic in MainCallback to meet this requirement:
```python
class CustomMainCB(MainCallBack):
    def on_iter_forward(self, runner):
        # Forward features
        image, target = runner.batch
        runner.output= runner.model(image, target)
        # Compute loss
        runner.loss = sum(loss for loss in runner.output.values())
```

Note that you must assign attributes `output` and `loss` back in `on_iter_forward`.

### Usage 2
In some cases user manually adjusts the learning rate without the help of the scheduler like:
```python
def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
```

We can implement the following logic in MainCallback to meet this requirement:
```python
class CustomMainCB(MainCallBack):
    def on_lr_adjust(self, runner):
        warmup_epoch = 5
        if runner.epochs <= warmup_epoch:
            lr = 1e-6 + (initial_lr-1e-6) * runner.batch_idx / (epoch_size * warmup_epoch)
        else:
            lr = initial_lr * (gamma ** (step_index))
        for param_group in runner.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
```

### Usage 3
If you want to save the currently trained model at the end of each epoch automatically, you may just implement this in `after_train_epoch`:
```python
class ModelCheckpoint(Callback):
    
    ...
    
    def after_train_epoch(self, runner):
        """
        Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        :param epoch:  Integer, index of epoch.
        """
        stats = {"epoch": runner.epochs}
        last_ckpt_path = self._format_checkpoint_name(dirname=self.dirname,
                                                      filename=self.filename,
                                                      stats=stats)
        runner.save_checkpoint(last_ckpt_path, self.save_weights_only)
```

Actually you can just pass `ModelCheckpoint` callback in `bigdl.orca.learn.pytorch.callbacks.model_checkpoint` to estimator.fit() to achieve this.
