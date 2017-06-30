**Scala:**
```scala
val optimMethod = new LBFGS(maxIter=20, maxEval=Double.MaxValue,
                            tolFun=1e-5, tolX=1e-9, nCorrection=100,
                            learningRate=1.0, lineSearch=None, lineSearchOptions=None)
```

**Python:**
```python
optim_method = LBFGS(max_iter=20, max_eval=Double.MaxValue, \
                 tol_fun=1e-5, tol_x=1e-9, n_correction=100, \
                 learning_rate=1.0, line_search=None, line_search_options=None)
```

This implementation of L-BFGS relies on a user-provided line search function
(state.lineSearch). If this function is not provided, then a simple learningRate
is used to produce fixed size steps. Fixed size steps are much less costly than line
searches, and can be useful for stochastic problems.

The learning rate is used even when a line search is provided.This is also useful for
large-scale stochastic problems, where opfunc is a noisy approximation of f(x). In that
case, the learning rate allows a reduction of confidence in the step size.

**Parameters:**
* **maxIter** - Maximum number of iterations allowed. Default: 20
* **maxEval** - Maximum number of function evaluations. Default: Double.MaxValue
* **tolFun** - Termination tolerance on the first-order optimality. Default: 1e-5
* **tolX** - Termination tol on progress in terms of func/param changes. Default: 1e-9
* **learningRate** - the learning rate. Default: 1.0
* **lineSearch** - A line search function. Default: None
* **lineSearchOptions** - If no line search provided, then a fixed step size is used. Default: None

**Scala example:**
```scala
val optimMethod = new LBFGS(maxIter=20, maxEval=Double.MaxValue,
                            tolFun=1e-5, tolX=1e-9, nCorrection=100,
                            learningRate=1.0, lineSearch=None, lineSearchOptions=None)
optimizer.setOptimMethod(optimMethod)
```

**Python example:**
```python
optim_method = LBFGS(max_iter=20, max_eval=DOUBLEMAX, \
                 tol_fun=1e-5, tol_x=1e-9, n_correction=100, \
                 learning_rate=1.0, line_search=None, line_search_options=None)
                  
optimizer = Optimizer(
    model=mlp_model,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=optim_method,
    end_trigger=MaxEpoch(20),
    batch_size=32)
```

