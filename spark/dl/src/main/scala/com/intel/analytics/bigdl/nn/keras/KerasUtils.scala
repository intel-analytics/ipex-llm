/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object KerasUtils {

  private[keras] def getPadsFromBorderMode(borderMode: String = "valid"): (Int, Int) = {
    if (borderMode == "same") {
      // padH, padW
      (-1, -1)
    } else {
      (0, 0)
    }
  }

  private[bigdl] def getInitMethod(init: String): InitializationMethod = {
    init.toLowerCase() match {
      case "glorot_uniform" => Xavier
      case "one" => Ones
      case "zero" => Zeros
      case "uniform" => RandomUniform(-0.05, 0.05)
      case "normal" => RandomNormal(0.0, 0.05)
      case _ => throw new IllegalArgumentException(s"Unsupported initialization method: " +
        s"${init.toLowerCase()}")
    }
  }

  private[bigdl] def getKerasActivation[T : ClassTag] (activation: String)
    (implicit ev: TensorNumeric[T]): KerasLayer[Tensor[T], Tensor[T], T] = {
    if (activation == null) { return null }
    if (activation.toLowerCase() == "softmax") {
      SoftMax[T]()
    } else {
      val torchActivation = getTorchActivation(activation)
      new KerasIdentityWrapper[T](torchActivation)
        .asInstanceOf[KerasLayer[Tensor[T], Tensor[T], T]]
    }
  }

  private[keras] def getTorchActivation[T : ClassTag] (activation: String)
    (implicit ev: TensorNumeric[T]): AbstractModule[Tensor[T], Tensor[T], T] = {
    if (activation == null) null
    else {
      activation.toLowerCase() match {
          case "tanh" => Tanh[T]()
          case "sigmoid" => Sigmoid[T]()
          case "relu" => ReLU[T]()
          case "softmax" =>
                com.intel.analytics.bigdl.nn.SoftMax[T]()
          case "softplus" => SoftPlus[T]()
          case "softsign" => SoftSign[T]()
          case "hard_sigmoid" => HardSigmoid[T]()
          case _ => throw new IllegalArgumentException(s"Invalid activation: " +
            s"${activation.toLowerCase}. Only simple activations can be constructed using string")
      }
    }
  }

  private[keras] def computeConvOutputLength(
    inputLength: Int,
    filterSize: Int,
    borderMode: String,
    stride: Int,
    dilation: Int = 1): Int = {
    val dilatedFilterSize = filterSize + (filterSize - 1) * (dilation - 1)
    val outputLength = borderMode match {
      case "valid" => inputLength - dilatedFilterSize + 1
      case "same" => inputLength
    }
    (outputLength + stride - 1) / stride
  }

  private[keras] def getPadsFromBorderMode3D(
    borderMode: String = "valid"): (Int, Int, Int) = {
    if (borderMode == "same") {
      // padT, padH, padW
      (-1, -1, -1)
    } else {
      (0, 0, 0)
    }
  }

  private[bigdl] def toBigDLFormat(dimOrdering: String): DataFormat = {
    require(dimOrdering.toLowerCase() == "tf" || dimOrdering.toLowerCase() == "th",
      s"Dim ordering must be either tf or th, but got ${dimOrdering.toLowerCase()}")
    dimOrdering.toLowerCase() match {
      case "tf" => DataFormat.NHWC
      case "th" => DataFormat.NCHW
    }
  }

  private[bigdl] def toBigDLFormat5D(dimOrdering: String): String = {
    require(dimOrdering.toLowerCase() == "tf" || dimOrdering.toLowerCase() == "th",
      s"Dim ordering must be either tf or th, but got ${dimOrdering.toLowerCase()}")
    dimOrdering.toLowerCase() match {
      case "tf" => "CHANNEL_LAST"
      case "th" => "CHANNEL_FIRST"
    }
  }

  private[keras] def toBigDLCriterion[T : ClassTag](loss: String)
    (implicit ev: TensorNumeric[T]): Criterion[T] = {
    loss.toLowerCase() match {
      case "binary_crossentropy" => BCECriterion[T]()
      case "categorical_crossentropy" => CategoricalCrossEntropy[T]()
      case "mse" => MSECriterion[T]()
      case "mean_squared_error" => MSECriterion[T]()
      case "mae" => AbsCriterion[T]()
      case "mean_absolute_error" => AbsCriterion[T]()
      case "hinge" => MarginCriterion[T]()
      case "mape" => MeanAbsolutePercentageCriterion[T]()
      case "mean_absolute_percentage_error" => MeanAbsolutePercentageCriterion[T]()
      case "msle" => MeanSquaredLogarithmicCriterion[T]()
      case "mean_squared_logarithmic_error" => MeanSquaredLogarithmicCriterion[T]()
      case "squared_hinge" => MarginCriterion[T](squared = true)
      case "sparse_categorical_crossentropy" => ClassNLLCriterion[T](logProbAsInput = false)
      case "kld" => KullbackLeiblerDivergenceCriterion[T]()
      case "kullback_leibler_divergence" => KullbackLeiblerDivergenceCriterion[T]()
      case "cosine_proximity" => CosineProximityCriterion[T]()
      case "poisson" => PoissonCriterion[T]()
      case _ => throw new IllegalArgumentException(s"Invalid loss: ${loss.toLowerCase()}")
    }
  }

  private[keras] def toBigDLOptimMethod[T: ClassTag](optimMethod: String)
    (implicit ev: TensorNumeric[T]): OptimMethod[T] = {
    optimMethod.toLowerCase() match {
      case "sgd" => new SGD[T](learningRate = 0.01)
      case "rmsprop" => new RMSprop[T](learningRate = 0.001, decayRate = 0.9)
      case "adamax" => new Adamax[T](Epsilon = 1e-8)
      case "adagrad" => new Adagrad[T](learningRate = 0.01)
      case "adadelta" => new Adadelta[T](decayRate = 0.95, Epsilon = 1e-8)
      case "adam" => new Adam[T]()
    }
  }

  private[keras] def toBigDLMetrics[T: ClassTag](metrics: Array[String])
    (implicit ev: TensorNumeric[T]): Array[ValidationMethod[T]] = {
    if (metrics == null) {
      null
    }
    else if (metrics.sameElements(Array("accuracy"))) {
      Array(new Top1Accuracy[T]())
    }
    else {
      throw new IllegalArgumentException(s"Unsupported metrics: ${metrics.mkString(", ")}")
    }
  }

}
