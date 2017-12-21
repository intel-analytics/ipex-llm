# **Summary**
Up to now, we have generally supported __ALL__ the layers of [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) to be loaded into BigDL.

Self-defined Keras layers or [`Lambda`](https://faroit.github.io/keras-docs/1.2.2/layers/core/#lambda) layers are not supported for now. Weight sharing is not supported for now.

This page lists the so-far unsupported arguments for specific layers and some known issues.

## **Unsupported Layer Arguments**
For the following arguments, currently only the default values are expected and supported.

* Constraints (`W_constraint`, `b_constraint`, etc.) are not supported for all layers.
* `activity_regularizer` is not supported for all layers.
* For [`Dropout`](https://faroit.github.io/keras-docs/1.2.2/layers/core/#dropout), `noise_shape` is not supported.
* For [`Merge`](https://faroit.github.io/keras-docs/1.2.2/layers/core/#merge) layers, `node_indices`, `tensor_indices`, `output_mask` are not supported. `Lambda/function` as merge mode is not supported.
* For [`Merge`](https://faroit.github.io/keras-docs/1.2.2/layers/core/#merge) layers with mode `'dot'` or `'cosine'`, only `2D` input with `dot_axes=1` is supported.
* For [`AtrousConvolution1D`](https://faroit.github.io/keras-docs/1.2.2/layers/convolutional/#atrousconvolution1d), only `bias=True` is supported.
* For [`AtrousConvolution2D`](https://faroit.github.io/keras-docs/1.2.2/layers/convolutional/#atrousconvolution2d), only `dim_ordering='th'` and `bias=True` are supported.
* For [`DeConvolution2D`](https://faroit.github.io/keras-docs/1.2.2/layers/convolutional/#deconvolution2d), only `dim_ordering='th'` is supported. For `border_mode='same'`, only symmetric padding on both sides is supported.
* For [`Convolution3D`](https://faroit.github.io/keras-docs/1.2.2/layers/convolutional/#convolution3d), only `dim_ordering='th'` is supported.
* For [`UpSampling3D`](https://faroit.github.io/keras-docs/1.2.2/layers/convolutional/#upsampling3d), only `dim_ordering='th'` is supported.
* For [`MaxPooling3D`](https://faroit.github.io/keras-docs/1.2.2/layers/pooling/#maxpooling3d), only `dim_ordering='th'` and `border_mode='valid'` are supported.
* For [`AveragePooling3D`](https://faroit.github.io/keras-docs/1.2.2/layers/pooling/#averagepooling3d), only `dim_ordering='th'` and `border_mode='valid'` are supported.
* For `GlobalMaxPooling3D`, only `dim_ordering='th'` is supported.
* For `GlobalAveragePooling3D`, only `dim_ordering='th'` is supported.
* For all [`Recurrent`](https://faroit.github.io/keras-docs/1.2.2/layers/recurrent/#recurrent) layers ([`SimpleRNN`](https://faroit.github.io/keras-docs/1.2.2/layers/recurrent/#simplernn), [`GRU`](https://faroit.github.io/keras-docs/1.2.2/layers/recurrent/#gru) and [`LSTM`](https://faroit.github.io/keras-docs/1.2.2/layers/recurrent/#lstm)), `stateful`, `consume_less`, `dropout_W` and `dropout_U` are not supported.
If RNNs are wrapped with [`Bidirectional`](https://faroit.github.io/keras-docs/1.2.2/layers/wrappers/#bidirectional), only `return_sequences=True` is supported.
* For [`Embedding`](https://faroit.github.io/keras-docs/1.2.2/layers/embeddings/#embedding), `mask_zero` and `dropout` are not supported.
* For [`PReLU`](https://faroit.github.io/keras-docs/1.2.2/layers/advanced-activations/#prelu), `init`, `weights` and `shared_axes` are not supported.
* For [`ParametricSoftplus`](https://faroit.github.io/keras-docs/1.2.2/layers/advanced-activations/#parametricsoftplus), `weights` and `shared_axes` are not supported. Only `alpha_init = 1/beta_init` is supported.
* For [`BatchNormalization`](https://faroit.github.io/keras-docs/1.2.2/layers/normalization/#batchnormalization), only `mode=0` is supported. Only `channel_first` (`dim_ordering='th'` with `axis=1`) and `channel_last` (`dim_ordering='tf'` with `axis=-1`) is supported. `gamma_regularizer` and `beta_regularizer` are not supported.


## **Known Issues**
* For some layers such as `ZeroPadding2D`, `ZeroPadding3D`, `Cropping2D`, `Cropping3D`, etc., Keras doesn't save `dim_ordering` into JSON. In this case, the default `dim_ordering` found in `~/.keras/keras.json` will be used instead.