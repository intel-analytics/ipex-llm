A loss function (or objective function), specified when you [compile](training/#compile) the model, is the function that the model intends to optimize in the process of training.

See [here](../../APIGuide/Losses/) for available loss objects.

For the sake of convenience, you can also use the corresponding string representation of a loss.

---
## **Available Losses**
* [mean_squared_error](../../APIGuide/Losses/#msecriterion) or [mse](../../APIGuide/Losses/#msecriterion)
* [mean_absolute_error](../../APIGuide/Losses/#abscriterion) or [mae](../../APIGuide/Losses/#abscriterion)
* [categorical_crossentropy](../../APIGuide/Losses/#categoricalcrossentropy)
* [sparse_categorical_crossentropy](../../APIGuide/Losses/#classnllcriterion)
* [binary_crossentropy](../../APIGuide/Losses/#bcecriterion)
* [mean_absolute_percentage_error](../../APIGuide/Losses/#meanabsolutepercentagecriterion) or [mape](../../APIGuide/Losses/#meanabsolutepercentagecriterion)
* [mean_squared_logarithmic_error](../../APIGuide/Losses/#meansquaredlogarithmiccriterion) or [msle](../../APIGuide/Losses/#meansquaredlogarithmiccriterion)
* [kullback_leibler_divergence](../../APIGuide/Losses/#kullbackleiblerdivergencecriterion) or [kld](../../APIGuide/Losses/#kullbackleiblerdivergencecriterion)
* [hinge](../../APIGuide/Losses/#margincriterion)
* [squared_hinge](../../APIGuide/Losses/#margincriterion)
* [poisson](../../APIGuide/Losses/#poissoncriterion)
* [cosine_proximity](../../APIGuide/Losses/#cosineproximitycriterion)
