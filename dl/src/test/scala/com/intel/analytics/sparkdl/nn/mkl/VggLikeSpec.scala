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

package com.intel.analytics.sparkdl.nn.mkl

import com.intel.analytics.sparkdl.nn
import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.optim.SGD
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag
object VggLikeBlas {
  def apply[T: ClassTag](classNum: Int)(implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
    val vggBnDo = new Sequential[Tensor[T], Tensor[T], T]()
    def convBNReLU(nInputPlane: Int, nOutPutPlane: Int): Sequential[Tensor[T], Tensor[T], T] = {
      vggBnDo.add(
        new nn.SpatialConvolution[T](nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1)
          .setInitMethod(Constant))
      vggBnDo.add(new nn.SpatialBatchNormalization[T](nOutPutPlane, 1e-3))
      vggBnDo.add(new nn.ReLU[T](false))
      vggBnDo
    }
    convBNReLU(3, 64).add(new Dropout[T]((0.3)))
    convBNReLU(64, 64)
    vggBnDo.add(new nn.SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLU(64, 128).add(new Dropout[T](0.4))
    convBNReLU(128, 128)
    vggBnDo.add(new nn.SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLU(128, 256).add(new Dropout[T](0.4))
    convBNReLU(256, 256).add(new Dropout[T](0.4))
    convBNReLU(256, 256)
    vggBnDo.add(new nn.SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLU(256, 512).add(new Dropout[T](0.4))
    convBNReLU(512, 512).add(new Dropout[T](0.4))
    convBNReLU(512, 512)
    vggBnDo.add(new nn.SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLU(512, 512).add(new Dropout[T](0.4))
    convBNReLU(512, 512).add(new Dropout[T](0.4))
    convBNReLU(512, 512)
    vggBnDo.add(new nn.SpatialMaxPooling[T](2, 2, 2, 2).ceil())
    vggBnDo.add(new View[T](512))

    val classifier = new Sequential[Tensor[T], Tensor[T], T]()
    classifier.add(new Dropout[T](0.5))
    classifier.add(new nn.Linear[T](512, 512))
    classifier.add(new nn.BatchNormalization[T](512))
    classifier.add(new nn.ReLU[T](true))
    classifier.add(new Dropout[T](0.5))
    classifier.add(new nn.Linear[T](512, classNum))
    classifier.add(new LogSoftMax[T])
    vggBnDo.add(classifier)

    println(vggBnDo)
    vggBnDo
  }
}

object VggLikeDnn {
  def apply[T: ClassTag](classNum: Int)(implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
    val vggBnDo = new Sequential[Tensor[T], Tensor[T], T]()
    def convBNReLUBN(nInputPlane: Int, nOutPutPlane: Int): Sequential[Tensor[T], Tensor[T], T] = {
      vggBnDo.add(new SpatialConvolution[T](nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1)
                    .setInitMethod(Constant))
      vggBnDo.add(new mkl.SpatialBatchNormalization[T](nOutPutPlane, 1e-3))
      vggBnDo.add(new ReLU[T](false))
      vggBnDo
    }

    def convBNReLU(nInputPlane: Int, nOutPutPlane: Int): Sequential[Tensor[T], Tensor[T], T] = {
      vggBnDo.add(new nn.SpatialConvolution[T](nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1)
          .setInitMethod(Constant))
      vggBnDo.add(new mkl.SpatialBatchNormalization[T](nOutPutPlane, 1e-3))
      vggBnDo.add(new nn.ReLU[T](false))
      vggBnDo
    }

    def convBNReLUNN(nInputPlane: Int, nOutPutPlane: Int): Sequential[Tensor[T], Tensor[T], T] = {
      vggBnDo.add(new nn.SpatialConvolution[T](nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1)
          .setInitMethod(Constant))
      vggBnDo.add(new mkl.SpatialBatchNormalization[T](nOutPutPlane, 1e-3))
      vggBnDo.add(new nn.ReLU[T](false))
      vggBnDo
    }
    convBNReLUBN(3, 64).add(new Dropout[T]((0.3)))
    convBNReLUBN(64, 64)
    vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLUBN(64, 128).add(new Dropout[T](0.4))
    convBNReLUBN(128, 128)
    vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLU(128, 256).add(new Dropout[T](0.4))
    convBNReLU(256, 256).add(new Dropout[T](0.4))
    convBNReLU(256, 256)
    vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLU(256, 512).add(new Dropout[T](0.4))
    convBNReLU(512, 512).add(new Dropout[T](0.4))
    convBNReLU(512, 512)
    vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLUNN(512, 512).add(new Dropout[T](0.4))
    convBNReLUNN(512, 512).add(new Dropout[T](0.4))
    convBNReLUNN(512, 512)
    vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())
    vggBnDo.add(new View[T](512))

    val classifier = new Sequential[Tensor[T], Tensor[T], T]()
    classifier.add(new Dropout[T](0.5))
    classifier.add(new nn.Linear[T](512, 512))
    classifier.add(new mkl.BatchNormalization[T](512))
    classifier.add(new nn.ReLU[T](true))
    classifier.add(new Dropout[T](0.5))
    classifier.add(new nn.Linear[T](512, classNum))
    classifier.add(new LogSoftMax[T])
    vggBnDo.add(classifier)

    println(vggBnDo)
    vggBnDo
  }
}

class VggLikeSpec extends FlatSpec with Matchers {
//  "VggLkie generete output and gradient" should "correctly" in {
//    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]) {
//      val batchSize = 4
//      val modelDnn = VggLikeDnn(10)
//      val modelBlas = VggLikeBlas(10)
//      val seqDnn = modelDnn.asInstanceOf[Sequential[T]]
//      val seqBlas = modelBlas.asInstanceOf[Sequential[T]]
//
//      modelDnn.reset()
//      modelBlas.reset()
//      val paraDnn = modelDnn.parameters()
//      val paraBlas = modelBlas.parameters()
//
//      for (i <- 0 until paraDnn._1.length) {
//        paraDnn._1(i).copy(paraBlas._1(i))
//      }
//
//      modelDnn.zeroGradParameters()
//      modelBlas.zeroGradParameters()
//
//      val input = Tensor[T](Array(batchSize, 3, 32, 32)).randn()
//
//      val criterionBlas = new ClassNLLCriterion[T]()
//      val labelsBlas = Tensor[T](batchSize).fill(ev.fromType(1))
//      val criterionDnn = new ClassNLLCriterion[T]()
//      val labelsDnn = Tensor[T](batchSize).fill(ev.fromType(1))
//
//      val sgdBlas = new SGD[T]()
//      val sgdDnn = new SGD[T]()
//
//      val stateBlas = T(
//        "learningRate" -> 0.01,
//        "weightDecay" -> 0.0005,
//        "momentum" -> 0.9,
//        "dampening" -> 0.0
//      )
//
//      val stateDnn = T(
//        "learningRate" -> 0.01,
//        "weightDecay" -> 0.0005,
//        "momentum" -> 0.9,
//        "dampening" -> 0.0
//      )
//
//      for (i <- 0 until Tools.getRandTimes()) {
//        val outputBlas = modelBlas.forward(input)
//        val errorBlas = criterionBlas.forward(outputBlas, labelsBlas)
//        val gradOutputBlas = criterionBlas.backward(outputBlas, labelsBlas)
//        val gradInputBlas = modelBlas.backward(input, gradOutputBlas)
//
//        val outputDnn = modelDnn.forward(input)
//        val errorDnn = criterionDnn.forward(outputDnn, labelsDnn)
//        val gradOutputDnn = criterionDnn.backward(outputDnn, labelsDnn)
//        val gradInputDnn = modelDnn.backward(input, gradOutputDnn)
//
////        for (i <- 0 until seqBlas.modules.length) {
////          val moduleName = seqDnn.modules(i).getName()
////          Tools.cumulativeError(seqDnn.modules(i).output,
////                                seqBlas.modules(i).output,
////                                ("module", moduleName, i, "output").productIterator.mkString(" "))
////        }
////
////        Tools.averageAll(gradInputDnn, "gradInput")
////        Tools.averageAll(outputDnn, "output")
//        Tools.cumulativeError(outputDnn, outputBlas, "iteration " + i + " output")
//        Tools.cumulativeError(gradOutputBlas, gradOutputDnn, "iteration " + i + " gradoutput")
//        Tools.cumulativeError(gradInputBlas, gradInputDnn, "iteration " + i + " gradinput")
//
//        val (weightsBlas, gradBlas) = modelBlas.getParameters()
//        val (weightsDnn, gradDnn) = modelDnn.getParameters()
//
//        sgdBlas.optimize(_ => (errorBlas, gradBlas), weightsBlas, stateBlas, stateBlas)
//        sgdDnn.optimize(_ => (errorDnn, gradDnn), weightsDnn, stateDnn, stateDnn)
//
//        Tools.cumulativeError(weightsBlas, weightsDnn,
//                              ("iteration", i, "weights").productIterator.mkString(" "))
//        Tools.cumulativeError(gradDnn, gradBlas,
//                              ("iteration", i, "gradient").productIterator.mkString(" "))
//        println("error Blas = " + errorBlas)
//        println("error Dnn = " + errorDnn)
//        println("for debug")
//      }
//
//      Tools.averageAllTensors(modelBlas.output, "blas output")
//      Tools.averageAllTensors(modelDnn.output, "dnn output")
//      Tools.cumulativeError(modelBlas.output, modelDnn.output,
//                            "output") should be(0.0 +- 1e-4)
//      Tools.averageAllTensors(modelBlas.gradInput, "blas gradinput")
//      Tools.averageAllTensors(modelDnn.gradInput, "dnn gradInput")
//      Tools.cumulativeError(modelDnn.gradInput, modelBlas.gradInput,
//                            "gradinput") should be(0.0 +- 2 * 1e-4)
//    }
//
//    test[Float]()
//  }
}
