## Sample

A `Sample` represent one record of your data set. One record contains feature and label, feature is one tensor or a few tensors; while label is one tensor or a few tensors, and it may be empty in testing or unsupervised learning. For example, one image and its category in image classification, one word in word2vec and one sentence and its label in RNN language model are all `Sample`.

Every Sample is actually a set of tensors, and them will be transformed to the input/output of the model. For example, in the case of image classification, a Sample have two tensors. One is 3D tensor representing a image, another is a 1-element tensor representing its category. For the 1-element label, you also can use a `T` instead of tensor.

** Scala example:
```scala
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val image = Tensor(3, 32, 32)
val label = 1f
val sample = Sample(image, label)
```

** Python example
```python
from bigdl.util.common import Sample
import numpy as np

image = np.random.rand(3, 32, 32)
label = np.array(1)
Sample.from_ndarray(image, label)
```

