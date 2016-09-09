/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.models

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.{T, Table}

import scala.reflect.ClassTag

object GoogleNet_v1 {
  private def inception[D: ClassTag](inputSize: Int, config: Table)(
    implicit ev: TensorNumeric[D]): Module[D] = {
    val concat = new Concat[D](2)
    val conv1 = new Sequential[D]
    conv1.add(new SpatialConvolution[D](inputSize,
      config[Table](1)(1), 1, 1, 1, 1).setInitMethod(Xavier))
    conv1.add(new ReLU[D](true))
    concat.add(conv1)
    val conv3 = new Sequential[D]
    conv3.add(new SpatialConvolution[D](inputSize,
      config[Table](2)(1), 1, 1, 1, 1).setInitMethod(Xavier))
    conv3.add(new ReLU[D](true))
    conv3.add(new SpatialConvolution[D](config[Table](2)(1),
      config[Table](2)(2), 3, 3, 1, 1, 1, 1).setInitMethod(Xavier))
    conv3.add(new ReLU[D](true))
    concat.add(conv3)
    val conv5 = new Sequential[D]
    conv5.add(new SpatialConvolution[D](inputSize,
      config[Table](3)(1), 1, 1, 1, 1).setInitMethod(Xavier))
    conv5.add(new ReLU[D](true))
    conv5.add(new SpatialConvolution[D](config[Table](3)(1),
      config[Table](3)(2), 5, 5, 1, 1, 2, 2).setInitMethod(Xavier))
    conv5.add(new ReLU[D](true))
    concat.add(conv5)
    val pool = new Sequential[D]
    pool.add(new SpatialMaxPooling[D](3, 3, 1, 1, 1, 1))
    pool.add(new SpatialConvolution[D](inputSize,
      config[Table](4)(1), 1, 1, 1, 1).setInitMethod(Xavier))
    concat.add(pool)
    concat
  }

  def apply[D: ClassTag](classNum: Int)(implicit ev: TensorNumeric[D]): Module[D] = {
    val model = new Sequential[D]
    model.add(new SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3).setInitMethod(Xavier))
    model.add(new ReLU[D](true))
    model.add(new SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    model.add(new LocalNormalizationAcrossChannels[D](5, 0.0001, 0.75))
    model.add(new SpatialConvolution[D](64, 64, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
    model.add(new ReLU[D](true))
    model.add(new SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier))
    model.add(new ReLU[D](true))
    model.add(new LocalNormalizationAcrossChannels[D](5, 0.0001, 0.75))
    model.add(new SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    model.add(inception[D](192, T(T(64), T(96, 128), T(16, 32), T(32))))
    model.add(inception[D](256, T(T(128), T(128, 192), T(32, 96), T(64))))
    model.add(new SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
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
  def apply[D: ClassTag](classNum: Int)(implicit ev: TensorNumeric[D]): Module[D] = {
    val features = new Sequential[D]
    features.add(new SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3))
    features.add(new SpatialBatchNormalization(64, 1e-3))
    features.add(new ReLU[D](true))
    features.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil())
    features.add(new SpatialConvolution[D](64, 64, 1, 1))
    features.add(new ReLU[D](true))
    features.add(new SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1))
    features.add(new SpatialBatchNormalization(192, 1e-3))
    features.add(new ReLU[D](true))
    features.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil())
    features.add(inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32))))
    features.add(inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64))))
    features.add(inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0))))
    features.add(new SpatialConvolution[D](576, 576, 2, 2, 2, 2))
    features.add(inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128))))
    features.add(inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128))))
    features.add(inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96))))
    features.add(inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96))))

    val mainBranch = new Sequential[D]
    mainBranch.add(inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0))))
    mainBranch.add(new SpatialConvolution[D](1024, 1024, 2, 2, 2, 2))
    mainBranch.add(new SpatialBatchNormalization(1024, 1e-3))
    mainBranch.add(inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128))))
    mainBranch.add(inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128))))
    mainBranch.add(new SpatialAveragePooling[D](7, 7, 1, 1))
    mainBranch.add(new View[D](1024).setNumInputDims(3))
    mainBranch.add(new Linear[D](1024, classNum))
    mainBranch.add(new LogSoftMax[D])

    val auxClassifier = new Sequential[D]
    auxClassifier.add(new SpatialAveragePooling[D](5, 5, 3, 3).ceil())
    auxClassifier.add(new SpatialConvolution[D](576, 128, 1, 1, 1, 1))
    auxClassifier.add(new SpatialBatchNormalization(128, 1e-3))
    auxClassifier.add(new View[D](128 * 4 * 4).setNumInputDims(3))
    auxClassifier.add(new Linear[D](128 * 4 * 4, 768))
    auxClassifier.add(new ReLU[D]())
    auxClassifier.add(new Linear[D](768, classNum))
    auxClassifier.add(new LogSoftMax[D])

    val splitter = new Concat[D](2)
    splitter.add(mainBranch)
    splitter.add(auxClassifier)

    val model = new Sequential[D]
    model.add(features)
    model.add(splitter)

    model
  }

  def inception[D: ClassTag](inputSize: Int, config: Table)(
    implicit ev: TensorNumeric[D]): Module[D] = {
    val concat = new Concat[D](2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = new Sequential[D]
      conv1.add(new SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1))
      conv1.add(new SpatialBatchNormalization(config[Table](1)(1), 1e-3))
      conv1.add(new ReLU[D](true))
      concat.add(conv1)
    }

    val conv3 = new Sequential[D]
    conv3.add(new SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1))
    conv3.add(new SpatialBatchNormalization(config[Table](2)(1), 1e-3))
    conv3.add(new ReLU[D](true))
    conv3.add(new SpatialConvolution[D](config[Table](2)(1),
      config[Table](2)(2), 3, 3, 1, 1, 1, 1))
    conv3.add(new SpatialBatchNormalization(config[Table](2)(2), 1e-3))
    conv3.add(new ReLU[D](true))
    concat.add(conv3)

    val conv3xx = new Sequential[D]
    conv3xx.add(new SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1))
    conv3xx.add(new SpatialBatchNormalization(config[Table](3)(1), 1e-3))
    conv3xx.add(new ReLU[D](true))

    conv3xx.add(new SpatialConvolution[D](config[Table](3)(1),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1))
    conv3xx.add(new SpatialBatchNormalization(config[Table](3)(2), 1e-3))
    conv3xx.add(new ReLU[D](true))

    conv3xx.add(new SpatialConvolution[D](config[Table](3)(2),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1))
    conv3xx.add(new SpatialBatchNormalization(config[Table](3)(2), 1e-3))
    conv3xx.add(new ReLU[D](true))
    concat.add(conv3xx)

    val pool = new Sequential[D]
    pool.add(new SpatialZeroPadding[D](1, 1, 1, 1))
    config[Table](4)[String](1) match {
      case "max" => pool.add(new SpatialMaxPooling[D](3, 3, 1, 1).ceil())
      case "avg" => pool.add(new SpatialAveragePooling[D](3, 3, 1, 1).ceil())
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(new SpatialConvolution[D](inputSize, config[Table](4)[Int](2), 1, 1, 1, 1))
      pool.add(new SpatialBatchNormalization(config[Table](4)(2), 1e-3))
      pool.add(new ReLU[D]())
    }
    concat.add(pool)

    concat
  }
}
