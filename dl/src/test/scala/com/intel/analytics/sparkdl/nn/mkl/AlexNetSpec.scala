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
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.sparkdl.utils.RandomGenerator._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/*
 * Note:
 *
 * 1. Dropout layer is deleted from all versions of model, because it
 *    is random.
 * 2. The output and gradInput cumulative error closes to 1e-4 ~ 1e-5,
 *    And the cumulative error depends on the input.
 */

object AlexNetBlas {
  def apply[T: ClassTag](classNum: Int)
                        (implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
    val model = new Sequential[Tensor[T], Tensor[T], T]()
    model.add(new SpatialConvolution[T](3, 96, 11, 11, 4, 4).setName("conv1")
                .setNeedComputeBack(false))
    model.add(
      new nn.SpatialConvolution[T](3, 96, 11, 11, 4, 4)
        .setName("conv1")
        .setNeedComputeBack(true)
        .setInitMethod(Xavier))
    model.add(new nn.ReLU[T](false).setName("relu1"))
    model.add(new nn.SpatialCrossMapLRN[T](5, 0.0001, 0.75).setName("norm1"))
    model.add(new nn.SpatialMaxPooling[T](3, 3, 2, 2).setName("pool1"))
    model.add(new nn.SpatialConvolution[T](96, 256, 5, 5, 1, 1, 2, 2, 1).setName("conv2"))
    model.add(new nn.ReLU[T](false).setName("relu2"))
    model.add(new nn.SpatialCrossMapLRN[T](5, 0.0001, 0.75).setName("norm2"))
    model.add(new nn.SpatialMaxPooling[T](3, 3, 2, 2).setName("pool2"))
    model.add(new nn.SpatialConvolution[T](256, 384, 3, 3, 1, 1, 1, 1).setName("conv3"))
    model.add(new nn.ReLU[T](false).setName("relu3"))
    model.add(new nn.SpatialConvolution[T](384, 384, 3, 3, 1, 1, 1, 1, 1).setName("conv4"))
    model.add(new nn.ReLU[T](false).setName("relu4"))
    model.add(new nn.SpatialConvolution[T](384, 256, 3, 3, 1, 1, 1, 1, 1).setName("conv5"))
    model.add(new nn.ReLU[T](false).setName("relu5"))
    model.add(new nn.SpatialMaxPooling[T](3, 3, 2, 2).setName("pool5"))
    model.add(new nn.View[T](256 * 6 * 6))
    model.add(new nn.Linear[T](256 * 6 * 6, 4096).setName("fc6"))
    model.add(new nn.ReLU[T](false).setName("relu6"))
    model.add(new nn.Dropout[T](0.5).setName("drop6"))
    model.add(new nn.Linear[T](4096, 4096).setName("fc7"))
    model.add(new nn.ReLU[T](false).setName("relu7"))
    model.add(new nn.Dropout[T](0.5).setName("drop7"))
    model.add(new nn.Linear[T](4096, classNum).setName("fc8"))
    model.add(new nn.LogSoftMax[T])
    model
  }
}

object AlexNetDnn {
  def apply[T: ClassTag](classNum: Int)
                        (implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
    val model = new Sequential[Tensor[T], Tensor[T], T]()
    model.add(
      new SpatialConvolution[T](3, 96, 11, 11, 4, 4)
        .setName("conv1")
        .setNeedComputeBack(true)
        .setInitMethod(Xavier))
    model.add(new ReLU[T](false).setName("relu1"))
    model.add(new SpatialCrossMapLRN[T](5, 0.0001, 0.75).setName("norm1"))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2).setName("pool1"))
    model.add(new SpatialConvolution[T](96, 256, 5, 5, 1, 1, 2, 2, 2).setName("conv2"))
    model.add(new ReLU[T](false).setName("relu2"))
    model.add(new SpatialCrossMapLRN[T](5, 0.0001, 0.75).setName("norm2"))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2).setName("pool2"))
    model.add(new SpatialConvolution[T](256, 384, 3, 3, 1, 1, 1, 1).setName("conv3"))
    model.add(new ReLU[T](false).setName("relu3"))
    model.add(new SpatialConvolution[T](384, 384, 3, 3, 1, 1, 1, 1, 2).setName("conv4"))
    model.add(new ReLU[T](false).setName("relu4"))
    model.add(new SpatialConvolution[T](384, 256, 3, 3, 1, 1, 1, 1, 2).setName("conv5"))
    model.add(new ReLU[T](false).setName("relu5"))
    model.add(new SpatialMaxPooling[T](3, 3, 2, 2).setName("pool5"))
    model.add(new View[T](256 * 6 * 6))
    model.add(new Linear[T](256 * 6 * 6, 4096).setName("fc6"))
    model.add(new ReLU[T](false).setName("relu6"))
    model.add(new Dropout[T](0.5).setName("drop6"))
    model.add(new Linear[T](4096, 4096).setName("fc7"))
    model.add(new ReLU[T](false).setName("relu7"))
    model.add(new Dropout[T](0.5).setName("drop7"))
    model.add(new Linear[T](4096, classNum).setName("fc8"))
    model.add(new Dummy[T]())
    model.add(new LogSoftMax[T]().setName("loss"))
    model
  }
}

class AlexNetSpec extends FlatSpec with Matchers {
/*  "AlexNet" should "generate correct output and gradient input" in {
    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val batchSize = 4
      val modelBlas = AlexNetBlas(100)
      val modelDnn = AlexNetDnn(100)

      modelBlas.reset()
      modelDnn.reset()

      RNG.setSeed(1000)

      val seqDnn = modelDnn.asInstanceOf[Sequential[T]]
      val seqBlas = modelBlas.asInstanceOf[Sequential[T]]

      val paraDnn = modelDnn.parameters()
      val paraBlas = modelBlas.parameters()

      for (i <- 0 until paraDnn._1.length) {
        paraBlas._1(i).copy(paraDnn._1(i))
      }

      val criterionBlas = new ClassNLLCriterion[T]()
      val labelsBlas = Tensor[T](batchSize).fill(ev.fromType(1))
      val criterionDnn = new ClassNLLCriterion[T]()
      val labelsDnn = Tensor[T](batchSize).fill(ev.fromType(1))

      val input = Tensor[T](Array(batchSize, 3, 227, 227)).rand()

      for (i <- 0 until Tools.getRandTimes()) {
        val outputBlas = modelBlas.forward(input)
        criterionBlas.forward(outputBlas, labelsBlas)
        val gradOutputBlas = criterionBlas.backward(outputBlas, labelsBlas)
        val gradInputBlas = modelBlas.backward(input, gradOutputBlas)

        val outputDnn = modelDnn.forward(input)
        criterionDnn.forward(outputDnn, labelsDnn)
        val gradOutputDnn = criterionDnn.backward(outputDnn, labelsDnn)
        val gradInputDnn = modelDnn.backward(input, gradOutputDnn)

        Tools.cumulativeError(outputDnn, outputBlas, "iteration " + i + " output")
        Tools.cumulativeError(gradOutputBlas, gradOutputDnn, "iteration " + i + " gradoutput")
        Tools.cumulativeError(gradInputBlas, gradInputDnn, "iteration " + i + " gradinput")
      }

      Tools.cumulativeError(modelBlas.output, modelDnn.output, "output") should be(0.0 +- 1e-5)
      Tools.cumulativeError(modelBlas.gradInput, modelDnn.gradInput, "gradinput") should be(
        0.0 +- 1e-4)
    }

    test[Float]()
  }*/

  "An AlexNet forward and backward" should "the same output, gradient as intelcaffe w/ dnn" in {
    val caffeCmd = Tools.getCollectCmd()
    val modelPath = Tools.getModuleHome() + "mkl2017_alexnet/train_val.prototxt"

    import scala.sys.process._
    (caffeCmd, modelPath).productIterator.mkString(" ").!!

    val batchSize = 4
    val model = AlexNetDnn[Float](1000)

    val criterion = new ClassNLLCriterion[Float]()
    // Attention, labels must be set to 1, or the value from caffe label + 1
    val labels = Tensor[Float](batchSize).fill(1)

    model.reset()
    val para = model.parameters()
    for (i <- 0 until para._1.length) {
      para._1(i).copy(Tools.getTensor[Float](f"CPUWght00$i%02d", para._1(i).size()))
    }
    val input = Tools.getTensor[Float]("CPUFwrd_data_input", Array(batchSize, 3, 227, 227))

    val modules = ArrayBuffer[TensorModule[Float]]()
    Tools.flattenModules(model, modules)

    val output = model.forward(input)
    val loss = criterion.forward(output, labels)
    val lossCaffe = Tools.getTensor[Float]("CPUFwrd_loss", Array(1))

    loss should be(lossCaffe.storage().array()(0))
/*

    val layerOutput = ArrayBuffer[Tensor[Float]]()
    for (i <- 0 until modules.length) {
      layerOutput += Tools.getTensorFloat("CPUFwrd_" + modules(i).getName().replaceAll("/", "_"),
                                      modules(i).output.size())

      Tools.cumulativeError(modules(i).output, layerOutput(i), "") should be (0.0)
    }
*/

    val gradOutput = criterion.backward(output, labels)
    val gradInput = model.backward(input, gradOutput)
/*

    val layerGradInput = ArrayBuffer[Tensor[Float]]()
    for (i <- 0 until modules.length) {
      layerGradInput += Tools.getTensorFloat("CPUBwrd_" + modules(i).getName().replaceAll("/", "_"),
                                             modules(i).output.size())
      Tools.cumulativeError(modules(i).gradInput, layerGradInput(i), "") should be (0.0)
    }
*/

    val gradInputCaffe = Tools.getTensor[Float]("CPUBwrd_conv1", gradInput.size())
    val gradWeightsCaffe = Tools.getTensor[Float]("CPUGrad0000", para._2(0).size())
/*

    val gradWeight = ArrayBuffer[Tensor[Float]]()
    for (i <- 0 until para._2.length) {
      gradWeight += Tools.getTensorFloat(f"CPUGrad00$i%02d", para._2(i).size())
      Tools.cumulativeError(para._2(i), gradWeight(i), "")
    }
*/
    Tools.cumulativeError(gradInput, gradInputCaffe, "gradInput") should be (0.0)
    Tools.cumulativeError(para._2(0), gradWeightsCaffe, "gradWeight") should be (0.0)
  }
}
