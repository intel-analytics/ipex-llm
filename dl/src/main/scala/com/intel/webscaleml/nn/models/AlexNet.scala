package com.intel.webscaleml.nn.models

import com.intel.webscaleml.nn.nn._
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
  * This is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
  */
object AlexNet_OWT {
  def apply[T : ClassTag](classNum : Int)(implicit ev: TensorNumeric[T]) : Module[T] = {
    val model = new Sequential[T]
    model.add(new SpatialConvolution[T](3, 64, 11, 11, 4, 4, 2, 2))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    model.add(new SpatialConvolution[T](64, 192, 5, 5, 1, 1, 2, 2))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    model.add(new SpatialConvolution[T](192, 384, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](384, 256, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    model.add(new View[T](256 * 6 * 6))
    model.add(new Dropout[T](0.5))
    model.add(new Linear[T](256 * 6 * 6, 4096))
    model.add(new Threshold[T](0, 1e-6))
    model.add(new Dropout[T](0.5))
    model.add(new Linear[T](4096, 4096))
    model.add(new Threshold[T](0, 1e-6))
    model.add(new Linear[T](4096, classNum))
    model.add(new LogSoftMax[T])
    model
  }
}

/**
  * ILSVRC2012 winner
  */
object AlexNet {
  def apply[T : ClassTag](classNum : Int)(implicit ev: TensorNumeric[T]) : Module[T] = {
    val features = new Concat[T](2)
    val fb1 = new Sequential[T]()
    fb1.add(new SpatialConvolution[T](3, 48, 11, 11, 4, 4, 2, 2))
    fb1.add(new ReLU[T](true))
    fb1.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    fb1.add(new SpatialConvolution[T](48, 128, 5, 5, 1, 1, 2, 2))
    fb1.add(new ReLU[T](true))
    fb1.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    fb1.add(new SpatialConvolution[T](128, 192, 3, 3, 1, 1, 1, 1))
    fb1.add(new ReLU[T](true))
    fb1.add(new SpatialConvolution[T](192, 192, 3, 3, 1, 1, 1, 1))
    fb1.add(new ReLU[T](true))
    fb1.add(new SpatialConvolution[T](192, 128, 3, 3, 1, 1, 1, 1))
    fb1.add(new ReLU[T](true))
    fb1.add(new SpatialMaxPooling[T](3, 3, 2, 2))

    val fb2 = new Sequential[T]()
    fb2.add(new SpatialConvolution[T](3, 48, 11, 11, 4, 4, 2, 2))
    fb2.add(new ReLU[T](true))
    fb2.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    fb2.add(new SpatialConvolution[T](48, 128, 5, 5, 1, 1, 2, 2))
    fb2.add(new ReLU[T](true))
    fb2.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    fb2.add(new SpatialConvolution[T](128, 192, 3, 3, 1, 1, 1, 1))
    fb2.add(new ReLU[T](true))
    fb2.add(new SpatialConvolution[T](192, 192, 3, 3, 1, 1, 1, 1))
    fb2.add(new ReLU[T](true))
    fb2.add(new SpatialConvolution[T](192, 128, 3, 3, 1, 1, 1, 1))
    fb2.add(new ReLU[T](true))
    fb2.add(new SpatialMaxPooling[T](3, 3, 2, 2))

    features.add(fb1)
    features.add(fb2)

    val model = new Sequential[T]()
    model.add(features)
    model.add(new View[T](256 * 6 * 6))
    model.add(new Dropout[T](0.5))
    model.add(new Linear[T](256 * 6 * 6, 4096))
    model.add(new Threshold[T](0, 1e-6))
    model.add(new Dropout[T](0.5))
    model.add(new Linear[T](4096, 4096))
    model.add(new Threshold[T](0, 1e-6))
    model.add(new Linear[T](4096, classNum))
    model.add(new LogSoftMax[T])
    model
  }
}
