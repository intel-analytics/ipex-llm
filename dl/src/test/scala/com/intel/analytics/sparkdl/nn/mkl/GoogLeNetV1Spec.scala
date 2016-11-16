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
import com.intel.analytics.sparkdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/*
 * TODO & Note:
 *
 * 1. For GoogLeNet v1, we should delete Dropout layer, because it random generate
 *    some data.
 * 2. Output and gradInput error cumulative error closes to 1e-5
 */

object GoogleNet_v1Blas {
  private def inception[D: ClassTag](inputSize: Int, config: Table, namePrefix: String)(
      implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val concat = new nn.Concat[D](2)
    val conv1 = new Sequential[Tensor[D], Tensor[D], D]
    conv1.add(
      new nn.SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName(namePrefix + "1x1"))
    conv1.add(new nn.ReLU[D](false).setName(namePrefix + "relu_1x1"))
    concat.add(conv1)
    val conv3 = new Sequential[Tensor[D], Tensor[D], D]
    conv3.add(
      new nn.SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName(namePrefix + "3x3_reduce"))
    conv3.add(new nn.ReLU[D](false).setName(namePrefix + "relu_3x3_reduce"))
    conv3.add(
      new nn.SpatialConvolution[D](config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName(namePrefix + "3x3"))
    conv3.add(new nn.ReLU[D](false).setName(namePrefix + "relu_3x3"))
    concat.add(conv3)
    val conv5 = new Sequential[Tensor[D], Tensor[D], D]
    conv5.add(
      new nn.SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName(namePrefix + "5x5_reduce"))
    conv5.add(new nn.ReLU[D](false).setName(namePrefix + "relu_5x5_reduce"))
    conv5.add(
      new nn.SpatialConvolution[D](config[Table](3)(1), config[Table](3)(2), 5, 5, 1, 1, 2, 2)
        .setInitMethod(Xavier)
        .setName(namePrefix + "5x5"))
    conv5.add(new nn.ReLU[D](false).setName(namePrefix + "relu_5x5"))
    concat.add(conv5)
    val pool = new Sequential[Tensor[D], Tensor[D], D]
    pool.add(new nn.SpatialMaxPooling[D](3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
    pool.add(
      new nn.SpatialConvolution[D](inputSize, config[Table](4)(1), 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName(namePrefix + "pool_proj"))
    pool.add(new nn.ReLU[D](false).setName(namePrefix + "relu_pool_proj"))
    concat.add(pool).setName(namePrefix + "output")
    concat
  }

  def apply[D: ClassTag](classNum: Int)(implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val feature1 = new Sequential[Tensor[D], Tensor[D], D]
    feature1.add(
      new nn.SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3)
        .setInitMethod(Xavier)
        .setName("conv1/7x7_s2")
        .setNeedComputeBack(true))
    feature1.add(new nn.ReLU[D](false).setName("conv1/relu_7x7"))
    feature1.add(new nn.SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    feature1.add(
      new nn.SpatialCrossMapLRN[D](5, 0.0001, 0.75).setName("pool1/norm1"))
    feature1.add(
      new nn.SpatialConvolution[D](64, 64, 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName("conv2/3x3_reduce"))
    feature1.add(new nn.ReLU[D](false).setName("conv2/relu_3x3_reduce"))
    feature1.add(
      new nn.SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName("conv2/3x3"))
    feature1.add(new nn.ReLU[D](false).setName("conv2/relu_3x3"))
    feature1.add(
      new nn.SpatialCrossMapLRN[D](5, 0.0001, 0.75).setName("conv2/norm2"))
    feature1.add(new nn.SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    feature1.add(inception[D](192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    feature1.add(inception[D](256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    feature1.add(new nn.SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool3/3x3_s2"))
    feature1.add(inception[D](480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))

    val output1 = new Sequential[Tensor[D], Tensor[D], D]
    output1.add(new nn.SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("loss1/ave_pool"))
    output1.add(new nn.SpatialConvolution[D](512, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(new nn.ReLU[D](false).setName("loss1/relu_conv"))
    output1.add(new View[D](128 * 4 * 4).setNumInputDims(3))
    output1.add(new nn.Linear[D](128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(new nn.ReLU[D](false).setName("loss1/relu_fc"))
    // output1.add(new Dropout[D](0.7).setName("loss1/drop_fc"))
    output1.add(new nn.Linear[D](1024, classNum).setName("loss1/classifier"))
    output1.add(new LogSoftMax[D].setName("loss1/loss"))

    val feature2 = new Sequential[Tensor[D], Tensor[D], D]
    feature2.add(inception[D](512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    feature2.add(inception[D](512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    feature2.add(inception[D](512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))

    val output2 = new Sequential[Tensor[D], Tensor[D], D]
    output2.add(new nn.SpatialAveragePooling[D](5, 5, 3, 3).setName("loss2/ave_pool"))
    output2.add(new nn.SpatialConvolution[D](528, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(new nn.ReLU[D](false).setName("loss2/relu_conv"))
    output2.add(new View[D](128 * 4 * 4).setNumInputDims(3))
    output2.add(new nn.Linear[D](128 * 4 * 4, 1024).setName("loss2/fc"))
    output2.add(new nn.ReLU[D](false).setName("loss2/relu_fc"))
    // output2.add(new Dropout[D](0.7).setName("loss2/drop_fc"))
    output2.add(new nn.Linear[D](1024, classNum).setName("loss2/classifier"))
    output2.add(new LogSoftMax[D].setName("loss2/loss"))

    val output3 = new Sequential[Tensor[D], Tensor[D], D]
    output3.add(inception[D](528, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_4e/"))
    output3.add(new nn.SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool4/3x3_s2"))
    output3.add(inception[D](832, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_5a/"))
    output3.add(inception[D](832, T(T(384), T(192, 384), T(48, 128), T(128)), "inception_5b/"))
    output3.add(new nn.SpatialAveragePooling[D](7, 7, 1, 1).setName("pool5/7x7_s1"))
    // output3.add(new nn.Dropout[D](0.4).setName("pool5/drop_7x7_s1"))
    output3.add(new View[D](1024).setNumInputDims(3))
    output3.add(new nn.Linear[D](1024, classNum).setInitMethod(Xavier).setName("loss3/classifier"))
    output3.add(new LogSoftMax[D].setName("loss3/loss3"))

    val split2 = new nn.Concat[D](2)
    split2.add(output3)
    split2.add(output2)

    val mainBranch = new Sequential[Tensor[D], Tensor[D], D]()
    mainBranch.add(feature2)
    mainBranch.add(split2)

    val split1 = new nn.Concat[D](2)
    split1.add(mainBranch)
    split1.add(output1)

    val model = new Sequential[Tensor[D], Tensor[D], D]()

    model.add(feature1)
    model.add(split1)

    model.reset()
    model
  }
}

object GoogleNet_v1Dnn {
  private def inception[D: ClassTag](inputSize: Int, config: Table, namePrefix: String)(
      implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val concat = new Concat[D](2)
    val conv1 = new Sequential[Tensor[D], Tensor[D], D]
    conv1.add(
      new SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName(namePrefix + "1x1"))
    conv1.add(new ReLU[D](false).setName(namePrefix + "relu_1x1"))
    concat.add(conv1)
    val conv3 = new Sequential[Tensor[D], Tensor[D], D]
    conv3.add(
      new SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName(namePrefix + "3x3_reduce"))
    conv3.add(new ReLU[D](false).setName(namePrefix + "relu_3x3_reduce"))
    conv3.add(
      new SpatialConvolution[D](config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName(namePrefix + "3x3"))
    conv3.add(new ReLU[D](false).setName(namePrefix + "relu_3x3"))
    concat.add(conv3)
    val conv5 = new Sequential[Tensor[D], Tensor[D], D]
    conv5.add(
      new SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName(namePrefix + "5x5_reduce"))
    conv5.add(new ReLU[D](false).setName(namePrefix + "relu_5x5_reduce"))
    conv5.add(
      new SpatialConvolution[D](config[Table](3)(1), config[Table](3)(2), 5, 5, 1, 1, 2, 2)
        .setInitMethod(Xavier)
        .setName(namePrefix + "5x5"))
    conv5.add(new ReLU[D](false).setName(namePrefix + "relu_5x5"))
    concat.add(conv5)
    val pool = new Sequential[Tensor[D], Tensor[D], D]
    pool.add(new SpatialMaxPooling[D](3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
    pool.add(
      new SpatialConvolution[D](inputSize, config[Table](4)(1), 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName(namePrefix + "pool_proj"))
    pool.add(new ReLU[D](false).setName(namePrefix + "relu_pool_proj"))
    concat.add(pool).setName(namePrefix + "output")
    concat
  }

  def apply[D: ClassTag](classNum: Int)(implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val feature1 = new Sequential[Tensor[D], Tensor[D], D]
    feature1.add(
      new SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3)
        .setInitMethod(Xavier)
        .setName("conv1/7x7_s2")
        .setNeedComputeBack(false))
    feature1.add(new ReLU[D](false).setName("conv1/relu_7x7"))
    feature1.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    feature1.add(new LocalNormalizationAcrossChannels[D](5, 0.0001, 0.75).setName("pool1/norm1"))
    feature1.add(
      new SpatialConvolution[D](64, 64, 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName("conv2/3x3_reduce"))
    feature1.add(new ReLU[D](false).setName("conv2/relu_3x3_reduce"))
    feature1.add(
      new SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1)
        .setInitMethod(Xavier)
        .setName("conv2/3x3"))
    feature1.add(new ReLU[D](false).setName("conv2/relu_3x3"))
    feature1.add(new LocalNormalizationAcrossChannels[D](5, 0.0001, 0.75).setName("conv2/norm2"))
    feature1.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    feature1.add(inception[D](192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    feature1.add(inception[D](256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    feature1.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool3/3x3_s2"))
    feature1.add(inception[D](480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))

    val output1 = new Sequential[Tensor[D], Tensor[D], D]
    output1.add(new SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("loss1/ave_pool"))
    output1.add(new SpatialConvolution[D](512, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(new ReLU[D](false).setName("loss1/relu_conv"))
    output1.add(new View[D](128 * 4 * 4).setNumInputDims(3))
    output1.add(new Linear[D](128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(new ReLU[D](false).setName("loss1/relu_fc"))
    // output1.add(new Dropout[D](0.7).setName("loss1/drop_fc"))
    output1.add(new Linear[D](1024, classNum).setName("loss1/classifier"))
    output1.add(new Dummy[D]())
    output1.add(new LogSoftMax[D].setName("loss1/loss"))

    val feature2 = new Sequential[Tensor[D], Tensor[D], D]
    feature2.add(inception[D](512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    feature2.add(inception[D](512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    feature2.add(inception[D](512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))

    val output2 = new Sequential[Tensor[D], Tensor[D], D]
    output2.add(new SpatialAveragePooling[D](5, 5, 3, 3).setName("loss2/ave_pool"))
    output2.add(new SpatialConvolution[D](528, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(new ReLU[D](false).setName("loss2/relu_conv"))
    output2.add(new View[D](128 * 4 * 4).setNumInputDims(3))
    output2.add(new Linear[D](128 * 4 * 4, 1024).setName("loss2/fc"))
    output2.add(new ReLU[D](false).setName("loss2/relu_fc"))
    // output2.add(new Dropout[D](0.7).setName("loss2/drop_fc"))
    output2.add(new Linear[D](1024, classNum).setName("loss2/classifier"))
    output2.add(new Dummy[D]())
    output2.add(new LogSoftMax[D].setName("loss2/loss"))

    val output3 = new Sequential[Tensor[D], Tensor[D], D]
    output3.add(inception[D](528, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_4e/"))
    output3.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool4/3x3_s2"))
    output3.add(inception[D](832, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_5a/"))
    output3.add(inception[D](832, T(T(384), T(192, 384), T(48, 128), T(128)), "inception_5b/"))
    output3.add(new SpatialAveragePooling[D](7, 7, 1, 1).setName("pool5/7x7_s1"))
    // output3.add(new Dropout[D](0.4).setName("pool5/drop_7x7_s1"))
    output3.add(new View[D](1024).setNumInputDims(3))
    output3.add(new Linear[D](1024, classNum).setInitMethod(Xavier).setName("loss3/classifier"))
    output3.add(new Dummy[D]())
    output3.add(new LogSoftMax[D].setName("loss3/loss3"))

    val split2 = new Concat[D](2)
    split2.add(output3)
    split2.add(output2)

    val mainBranch = new Sequential[Tensor[D], Tensor[D], D]()
    mainBranch.add(feature2)
    mainBranch.add(split2)

    val split1 = new Concat[D](2)
    split1.add(mainBranch)
    split1.add(output1)

    val model = new Sequential[Tensor[D], Tensor[D], D]()

    model.add(feature1)
    model.add(split1)

    model.reset()
    model
  }
}

class GoogLeNetV1Spec extends FlatSpec with Matchers {
  "GoogLeNet v1" should "generate correct result" in {
    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]) {
      val batchSize = 8
      val modelDnn = GoogleNet_v1Dnn(1000)
      val modelBlas = GoogleNet_v1Blas(1000)
//      val seqDnn = modelDnn.asInstanceOf[Sequential[T]]
//      val seqBlas = modelBlas.asInstanceOf[Sequential[T]]

      modelDnn.reset()
      modelBlas.reset()
      val paraDnn = modelDnn.parameters()
      val paraBlas = modelBlas.parameters()

      for (i <- 0 until paraDnn._1.length) {
        paraBlas._1(i).copy(paraDnn._1(i))
      }

      val input = Tensor[T](Array(batchSize, 3, 224, 224)).rand()

      val criterionBlas = new ClassNLLCriterion[T]()
      val labelsBlas = Tensor[T](batchSize).fill(ev.fromType(1))
      val criterionDnn = new ClassNLLCriterion[T]()
      val labelsDnn = Tensor[T](batchSize).fill(ev.fromType(1))

      for (i <- 0 until Tools.getRandTimes()) {
        val outputBlas = modelBlas.forward(input)
        criterionBlas.forward(outputBlas, labelsBlas)
        val gradOutputBlas = criterionBlas.backward(outputBlas, labelsBlas)
        val gradInputBlas = modelBlas.backward(input, gradOutputBlas)

        val outputDnn = modelDnn.forward(input)
        criterionDnn.forward(outputDnn, labelsDnn)
        val gradOutputDnn = criterionDnn.backward(outputDnn, labelsDnn)
        val gradInputDnn = modelDnn.backward(input, gradOutputDnn)

/*        for (i <- 0 until seqBlas.modules.length) {
          Tools.cumulativeError(seqDnn.modules(i).output.asInstanceOf[Tensor[T]],
                                seqBlas.modules(i).output.asInstanceOf[Tensor[T]],
                                "module " + i + " output")
        }
        for (i <- 0 until seqBlas.modules.length) {
          Tools.averageError(seqDnn.modules(i).output.asInstanceOf[Tensor[T]],
                             seqBlas.modules(i).output.asInstanceOf[Tensor[T]],
                             "module " + i + " output")
        }*/

        Tools.cumulativeError(outputDnn, outputBlas, "iteration " + i + " output")
        Tools.cumulativeError(gradOutputBlas, gradOutputDnn, "iteration " + i + " gradoutput")
        Tools.cumulativeError(gradInputBlas, gradInputDnn, "iteration " + i + " gradinput")

/*        val output1Dnn =
          modelDnn.asInstanceOf[Sequential[T]].modules(1).asInstanceOf[Concat[T]].modules(1)
        val output1Blas =
          modelBlas.asInstanceOf[Sequential[T]].modules(1).asInstanceOf[nn.Concat[T]].modules(1)

        Tools.cumulativeError(output1Dnn.output, output1Blas.output, "output1 " + i + " output")
        Tools.cumulativeError(output1Dnn.gradInput,
                              output1Blas.gradInput,
                              "output1 " + i + " gradinput")

        val output2Dnn = modelDnn
          .asInstanceOf[Sequential[T]]
          .modules(1)
          .asInstanceOf[Concat[T]]
          .modules(0)
          .asInstanceOf[Sequential[T]]
          .modules(1)
          .asInstanceOf[Concat[T]]
          .modules(1)
        val output2Blas = modelBlas
          .asInstanceOf[Sequential[T]]
          .modules(1)
          .asInstanceOf[nn.Concat[T]]
          .modules(0)
          .asInstanceOf[Sequential[T]]
          .modules(1)
          .asInstanceOf[nn.Concat[T]]
          .modules(1)

        Tools.cumulativeError(output2Dnn.output, output2Blas.output, "output2 " + i + " output")
        Tools.cumulativeError(output2Dnn.gradInput,
                              output2Blas.gradInput,
                              "output2 " + i + " gradinput")

        val output3Dnn = modelDnn
          .asInstanceOf[Sequential[T]]
          .modules(1)
          .asInstanceOf[Concat[T]]
          .modules(0)
          .asInstanceOf[Sequential[T]]
          .modules(1)
          .asInstanceOf[Concat[T]]
          .modules(0)
        val output3Blas = modelBlas
          .asInstanceOf[Sequential[T]]
          .modules(1)
          .asInstanceOf[nn.Concat[T]]
          .modules(0)
          .asInstanceOf[Sequential[T]]
          .modules(1)
          .asInstanceOf[nn.Concat[T]]
          .modules(0)

        Tools.cumulativeError(output3Dnn.output, output3Blas.output, "output3 " + i + " output")
        Tools.cumulativeError(output3Dnn.gradInput,
                              output3Blas.gradInput,
                              "output3 " + i + " gradinput")*/
      }

      Tools.averageAllTensors(modelBlas.output, "blas output")
      Tools.averageAllTensors(modelDnn.output, "dnn output")
      Tools.cumulativeError(modelBlas.output, modelDnn.output, "output") should be(0.0 +- 1e-4)
      Tools.averageAllTensors(modelBlas.gradInput, "blas gradinput")
      Tools.averageAllTensors(modelDnn.gradInput, "dnn gradInput")
      Tools.cumulativeError(modelDnn.gradInput, modelBlas.gradInput, "gradinput") should be(
        0.0 +- 1e-5)
    }

    test[Float]()
    // test[Double]()
  }
  "An AlexNet forward and backward" should "the same output, gradient as intelcaffe w/ dnn" in {
//    val caffeCmd = Tools.getCollectCmd()
//    val modelPath = Tools.getModuleHome() + "mkl2017_googlenet_v1_bdw/train_val.prototxt"
//
//    import scala.sys.process._
//    (caffeCmd, modelPath).productIterator.mkString(" ").!!

    val batchSize = 32
    val model = GoogleNet_v1Dnn[Float](1000)

    val criterion = new ClassNLLCriterion[Float]()
    // Attention, labels must be set to 1, or the value from caffe label + 1
    val labels = Tensor[Float](batchSize).fill(1)

    model.reset()
    val input = Tools.getTensor[Float]("CPUFwrd_data_input", Array(batchSize, 3, 224, 224))

    val modules = ArrayBuffer[TensorModule[Float]]()
    Tools.flattenModules(model, modules)

    for (i <- 0 until modules.length) {
      val para = modules(i).parameters()
      if (para != null) {
        for (j <- 0 until para._1.length) {
          val binName = "CPUFwrd_" + modules(i).getName().replaceAll("/", "_") + "Wght" + j
          para._1(j).copy(Tools.getTensor[Float](binName, para._1(j).size()))
        }
      }
    }

    val output = model.forward(input)
    val loss = criterion.forward(output, labels)
    val lossCaffe = Tools.getTensor[Float]("CPUFwrd_loss3_loss3", Array(1))

    val layerOutput = new Array[Tensor[Float]](modules.length)
    val layerGradInput = new Array[Tensor[Float]](modules.length)
    for (i <- 0 until modules.length) {
      layerOutput(i) = Tools.getTensor[Float]("CPUFwrd_" + modules(i).getName().replaceAll("/", "_"),
        modules(i).output.size())

//      Tools.cumulativeError(modules(i).output, layerOutput(i), modules(i).getName()) should be (0.0)
      if (layerOutput(i).nElement() > 0) {
        val error = Tools.cumulativeError(modules(i).output, layerOutput(i), modules(i).getName())
        if (error != 0) {
          val sb = modules(i-1).output
          val s = modules(i).output

          val cb = layerOutput(i-1)
          val c = layerOutput(i)

          println("calm down")
        }
      }
    }

    loss should be(lossCaffe.storage().array()(0))

    val gradOutput = criterion.backward(output, labels)
    val gradInput = model.backward(input, gradOutput)
    for (i <- modules.length - 1 to 0 by -1) {
      layerGradInput(i) = Tools.getTensor[Float]("CPUBwrd_" + modules(i).getName().replaceAll("/", "_"),
        modules(i).gradInput.size())

//      Tools.cumulativeError(modules(i).gradInput, layerOutput(i), modules(i).getName()) should be (0.0)
      if (layerGradInput(i).nElement() > 0) {
        if (Tools.cumulativeError(modules(i).gradInput, layerGradInput(i), modules(i).getName()) != 0) {
          val sb = if (i < modules.length - 1) modules(i + 1).gradInput else null
          val s = modules(i).gradInput

          val cb = if (i < modules.length - 1) layerGradInput(i + 1) else null
          val c = layerGradInput(i)

          println("calm down")
        }
      }
    }
    val firstLayerName = "CPUBwrd_" + modules(0).getName().replaceAll("/", "_")
    val gradInputCaffe = Tools.getTensor[Float](firstLayerName, gradInput.size())
    Tools.cumulativeError(gradInput, gradInputCaffe, "gradInput") should be (0.0)

    val para = modules(0).parameters()
    for (i <- 0 until para._2.length) {
      val binName = firstLayerName + "Grad" + i
      val gradCaffe = Tools.getTensor[Float](binName, para._2(i).size())
    }
  }
}
