# How to Customize Your Training Process

## Callback Mechanism Introduction

### What is callback mechanism?

A callback function is a function passed into another function as an argument, which is then invoked inside the outer function to complete some kind of routine or action.

### How does a callback work?

Callbacks mechanism divides the entire training process into several points according to different stages, so users can implement various processes of training by overriding these callbacks of corresponding points.

### How many callbacks are there?

And there are two kinds of callbacks here:

One is the callback that does nothing by default like `on_run_end`. We may utilize our existing Callback class to achieve this.
```python
class Callback:
    
    def on_run_begin(self, runner): ...
    def on_run_end(self, runner): ...
    def on_epoch_begin(self, runner): ...
    def on_epoch_end(self, runner): ...
    def on_iter_begin(self, runner): ...
    def on_iter_end(self, runner): ...
```

Another is the callback that contains default behaviors defined by orca: on_iter_forward, on_iter_backward and on_lr_adjust. These methods are somewhat special, because only one special MainCallback should be allowed to implement these methods among all callbacks(otherwise there will propagate forward and backward twice).
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

In general, there are 23 points where callbacks can be inserted from the beginning to the end of model training:

* global points: `before_run`, `after_run`, `before_epoch`, `after_epoch`, `before_iter`, `after_iter`, *`on_iter_forward`
* training points: `before_train_epoch`, `after_train_epoch`,  `before_train_iter`, `before_val_iter`, *`on_train_forward`, *`on_iter_backward`, *`on_lr_adjust`
* validation points: `before_val_epoch`, `after_val_epoch`, `before_val_iter`, `after_val_iter`, *`on_val_forward`
* prediction points: `before_pred_epoch`, `after_pred_epoch`, `before_pred_iter`, `after_pred_iter`

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

        call_hooks("on_iter_backward", self)# this will be called when training

        call_hooks("on_iter_end", self)

        # We should not present items in last iteration to users in the next iteration.
        if hasattrs(self, 'batch'|'batch_idx'|'outputs'|'loss')
           del self.batch
           del self.batch_idx
           del self.outputs
           del self.loss

    call_hooks("on_lr_adjust", self)# this will be called when training

    call_hooks("on_epoch_end", self)

call_hooks("on_run_end", self)

```
So you may write down your forwarding logic by **accessing these runner's attributes** like:
```python
class CustomMainCB(MainCallBack):
    def on_iter_forward(self, runner):
        # Forward features
        image, target = runner.batch
        runner.output= runner.model(image, target)
        # Compute loss
        runner.loss = sum(loss for loss in runner.output.values())
```
### What attributes user can access during the whole process?

There are 