## Adamax ##

An implementation of Adamax http://arxiv.org/pdf/1412.6980.pdf

Arguments:

* learningRate : learning rate
* beta1 : first moment coefficient
* beta2 : second moment coefficient
* Epsilon : for numerical stability

Returns:

the new x vector and the function list {fx}, evaluated before the update
