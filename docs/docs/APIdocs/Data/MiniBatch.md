## MiniBatch
MiniBatch` is a data structure to feed input/target to model in `Optimizer`. It provide `getInput()` and `getTarget` function to get the input and target in this MiniBatch.

`MiniBatch` can be created by `MiniBatch(nInputs: Int, nOutputs: Int)`, `nInputs` means number of inputs, `nOutputs` means number of outputs. And you can use `set(samples: Seq[Sample[T])` to fill the content in this MiniBatch. Like: 
