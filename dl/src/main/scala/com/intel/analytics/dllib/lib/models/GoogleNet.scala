package com.intel.analytics.dllib.lib.models

import com.intel.analytics.dllib.lib.nn._
import com.intel.analytics.dllib.lib.tensor.{T, Table}
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object GoogleNet_v1 {
  private def inception[D : ClassTag](inputSize: Int, config: Table)(implicit ev: TensorNumeric[D]): Module[D] = {
    val concat = new Concat[D](2)
    val conv1 = new Sequential[D]
    conv1.add(new SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1).setInitMethod(Xavier))
    conv1.add(new ReLU[D](true))
    concat.add(conv1)
    val conv3 = new Sequential[D]
    conv3.add(new SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1).setInitMethod(Xavier))
    conv3.add(new ReLU[D](true))
    conv3.add(new SpatialConvolution[D](config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1).setInitMethod(Xavier))
    conv3.add(new ReLU[D](true))
    concat.add(conv3)
    val conv5 = new Sequential[D]
    conv5.add(new SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1).setInitMethod(Xavier))
    conv5.add(new ReLU[D](true))
    conv5.add(new SpatialConvolution[D](config[Table](3)(1), config[Table](3)(2), 5, 5, 1, 1, 2, 2).setInitMethod(Xavier))
    conv5.add(new ReLU[D](true))
    concat.add(conv5)
    val pool = new Sequential[D]
    pool.add(new SpatialMaxPooling[D](3, 3, 1, 1, 1, 1))
    pool.add(new SpatialConvolution[D](inputSize, config[Table](4)(1), 1, 1, 1, 1).setInitMethod(Xavier))
    concat.add(pool)
    concat
  }

  def apply[D : ClassTag](classNum : Int)(implicit ev: TensorNumeric[D]) : Module[D] = {
    val model = new Sequential[D]
    model.add(new SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3).setInitMethod(Xavier))
    model.add(new ReLU[D](true))
    model.add(new SpatialMaxPooling[D](3,3,2,2,1,1))
    model.add(new LocalNormalizationAcrossChannels[D](5, 0.0001, 0.75))
    model.add(new SpatialConvolution[D](64,64,1,1,1,1,0,0).setInitMethod(Xavier))
    model.add(new ReLU[D](true))
    model.add(new SpatialConvolution[D](64,192,3,3,1,1,1,1).setInitMethod(Xavier))
    model.add(new ReLU[D](true))
    model.add(new LocalNormalizationAcrossChannels[D](5, 0.0001, 0.75))
    model.add(new SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    model.add(inception[D](192, T(T(64), T(96, 128), T(16, 32), T(32))))
    model.add(inception[D](256, T(T(128), T(128, 192), T(32, 96), T(64))))
    model.add(new SpatialMaxPooling[D](3,3,2,2,1,1))
    model.add(inception[D](480, T(T(192), T(96, 208), T(16, 48), T(64))))
    model.add(inception[D](512, T(T(160), T(112, 224), T(24, 64), T(64))))
    model.add(inception[D](512, T(T(128), T(128, 256), T(24, 64), T(64))))
    model.add(inception[D](512, T(T(112), T(144, 288), T(32, 64), T(64))))
    model.add(inception[D](528, T(T(256), T(160, 320), T(32, 128), T(128))))
    model.add(new SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    model.add(inception[D](832, T(T(256), T(160, 320), T(32, 128), T(128))))
    model.add(inception[D](832, T(T(384), T(192, 384), T(48, 128), T(128))))
    model.add(new SpatialAveragePooling[D](7, 7, 1, 1))
    model.add(new Dropout[D](0.4))
    model.add(new View[D](1024).setNumInputDims(3))
    model.add(new Linear[D](1024, classNum).setInitMethod(Xavier))
    model.add(new LogSoftMax[D])
    model.reset()
    model
  }
}

object GoogleNet_v2 {
  def apply[D : ClassTag](classNum : Int)(implicit ev: TensorNumeric[D]) : Unit = {

  }
}
