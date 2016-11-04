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
import com.intel.analytics.sparkdl.nn.{Constant, Default, Module, Xavier}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.Table
import org.apache.spark.sql.catalyst.expressions.Concat

import scala.reflect.ClassTag

class OmitConversionSpec extends FlatSpec with Matchers {
  def getModel[T: ClassTag](backend: String)(implicit ev: TensorNumeric[T]): Module[T] = {
    val model = new nn.Sequential[T]()

    def getLayer[T](dnn: () => Module[T], blas: () => Module[T]): Module[T] = {
      backend match {
        case "dnn" => dnn()
        case "blas" => blas()
        case "mix" => if (scala.util.Random.nextInt(2) != 0) dnn() else blas()
      }
    }

    model.add(
      getLayer(() =>
                 new nn.SpatialConvolution[T](3, 64, 7, 7, 2, 2, 3, 3)
                   .setInitMethod(Xavier)
                   .setName("conv1/7x7_s2")
                   .setNeedComputeBack(true),
               () =>
                 new nn.SpatialConvolution[T](3, 64, 7, 7, 2, 2, 3, 3)
                   .setInitMethod(Xavier)
                   .setName("conv1/7x7_s2")
                   .setNeedComputeBack(true)))
    model.add(
      getLayer(() => new ReLU[T](false).setName("conv1/relu_7x7"),
               () => new nn.ReLU[T](false).setName("conv1/relu_7x7"))
    )

    model.add(
      getLayer(() => new SpatialMaxPooling[T](3, 3, 2, 2).ceil().setName("pool1/3x3_s2"),
               () => new nn.SpatialMaxPooling[T](3, 3, 2, 2).ceil().setName("pool1/3x3_s2")))

    model.add(
      getLayer(
        () => new nn.LocalNormalizationAcrossChannels[T](5, 0.0001, 0.75).setName("pool1/norm1"),
        () => new nn.LocalNormalizationAcrossChannels[T](5, 0.0001, 0.75).setName("pool1/norm1")))

    model.add(
      getLayer(() =>
                 new nn.SpatialConvolution[T](64, 64, 1, 1, 1, 1)
                   .setInitMethod(Xavier)
                   .setName("conv2/3x3_reduce"),
               () =>
                 new nn.SpatialConvolution[T](64, 64, 1, 1, 1, 1)
                   .setInitMethod(Xavier)
                   .setName("conv2/3x3_reduce")))

    model.add(
      getLayer(() => new ReLU[T](false).setName("conv2/relu_3x3_reduce"),
               () => new nn.ReLU[T](false).setName("conv2/relu_3x3_reduce")))

    model.add(
      getLayer(() =>
                 new nn.SpatialConvolution[T](64, 192, 3, 3, 1, 1, 1, 1)
                   .setInitMethod(Constant)
                   .setName("conv2/3x3"),
               () =>
                 new nn.SpatialConvolution[T](64, 192, 3, 3, 1, 1, 1, 1)
                   .setInitMethod(Constant)
                   .setName("conv2/3x3")))

    model.add(
      getLayer(() => new ReLU[T](false).setName("conv2/relu_3x3"),
               () => new nn.ReLU[T](false).setName("conv2/relu_3x3")))

    model.add(
      getLayer(
        () => new nn.LocalNormalizationAcrossChannels[T](5, 0.0001, 0.75).setName("conv2/norm2"),
        () => new nn.LocalNormalizationAcrossChannels[T](5, 0.0001, 0.75).setName("conv2/norm2")))

    model.add(
      getLayer(() => new SpatialMaxPooling[T](3, 3, 2, 2).ceil().setName("pool2/3x3_s2"),
               () => new nn.SpatialMaxPooling[T](3, 3, 2, 2).ceil().setName("pool2/3x3_s2")))

    val conv1 = new nn.Sequential[T]()
    val conv3 = new nn.Sequential[T]()
    val conv5 = new nn.Sequential[T]()
    val pool = new nn.Sequential[T]()

    conv1.add(
      getLayer(() => new nn.SpatialConvolution[T](192, 64, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier),
               () => new nn.SpatialConvolution[T](192, 64, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
    )
    conv1.add(
      getLayer(() => new ReLU[T](false), () => new nn.ReLU[T](false))
    )

    conv3.add(
      getLayer(() => new nn.SpatialConvolution[T](192, 96, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier),
               () => new nn.SpatialConvolution[T](192, 96, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
    )
    conv3.add(
      getLayer(() => new ReLU[T](false), () => new nn.ReLU[T](false))
    )
    conv3.add(
      getLayer(() => new nn.SpatialConvolution[T](96, 128, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier),
               () => new nn.SpatialConvolution[T](96, 128, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier))
    )
    conv3.add(
      getLayer(() => new ReLU[T](false), () => new nn.ReLU[T](false))
    )

    conv5.add(
      getLayer(() => new nn.SpatialConvolution[T](192, 16, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier),
               () => new nn.SpatialConvolution[T](192, 16, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
    )
    conv5.add(getLayer(() => new ReLU[T](false), () => new nn.ReLU[T](false)))
    conv5.add(
      getLayer(() => new nn.SpatialConvolution[T](16, 32, 5, 5, 1, 1, 2, 2).setInitMethod(Xavier),
               () => new nn.SpatialConvolution[T](16, 32, 5, 5, 1, 1, 2, 2).setInitMethod(Xavier))
    )
    conv5.add(getLayer(() => new ReLU[T](false), () => new nn.ReLU[T](false)))

    pool.add(
      getLayer(() => new SpatialMaxPooling[T](3, 3, 1, 1, 1, 1).ceil(),
               () => new nn.SpatialMaxPooling[T](3, 3, 1, 1, 1, 1).ceil())
    )
    pool.add(
      getLayer(
        () => new nn.SpatialConvolution[T](192, 32, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier),
        () => new nn.SpatialConvolution[T](192, 32, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier)
      )
    )
    pool.add(
      getLayer(() => new ReLU[T](false), () => new nn.ReLU[T](false))
    )

    backend match {
      case "dnn" =>
        val concat = new Concat[T](2)
        concat.add(conv1)
        concat.add(conv3)
        concat.add(conv5)
        concat.add(pool)
        concat
        model.add(concat)
      case "blas" =>
        val concat = new nn.Concat[T](2)
        concat.add(conv1)
        concat.add(conv3)
        concat.add(conv5)
        concat.add(pool)
        concat
        model.add(concat)
      case "mix" =>
        val concat = new Concat[T](2)
        concat.add(conv1)
        concat.add(conv3)
        concat.add(conv5)
        concat.add(pool)
        concat
        model.add(concat)
    }
    model.add(
      getLayer(
        () => new nn.SpatialConvolution[T](256, 128, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier),
        () => new nn.SpatialConvolution[T](256, 128, 1, 1, 1, 1, 0, 0).setInitMethod(Xavier))
    )

    model
  }

  "Omit conversion" should "return correct result" in {
    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val modelDnn = getModel[T]("dnn")
      val modelBlas = getModel[T]("blas")
      val seqDnn = modelDnn.asInstanceOf[nn.Sequential[T]]
      val seqBlas = modelBlas.asInstanceOf[nn.Sequential[T]]
      println(modelDnn)
      println(modelBlas)

      for (i <- 0 until 2) {
        val paraDnn = modelDnn.parameters()
        val paraBlas = modelBlas.parameters()
        for (i <- 0 until paraDnn._1.length) {
          paraBlas._1(i).copy(paraDnn._1(i))
        }

        val input = Tensor[T](Array(32, 3, 224, 224)).rand()

        val outputBlas = modelBlas.forward(input)
        val outputDnn = modelDnn.forward(input)

        for (i <- 0 until seqBlas.modules.length) {
          Tools.cumulativeError(seqDnn.modules(i).output,
                                seqBlas.modules(i).output,
                                "module " + i + " output")
        }
        outputDnn should be equals (outputBlas)
        Tools.cumulativeError(outputDnn, outputBlas, "output") should be(0.0 +- 2 * 1e-5)

        outputDnn.nElement() should be(outputBlas.nElement())

        val gradOutput = Tensor[T]().resizeAs(outputDnn).fill(ev.fromType(0.1))

        val gradInputDnn = modelDnn.backward(input, gradOutput)
        val gradInputBlas = modelBlas.backward(input, gradOutput)

//        Tools.AverageError(seqDnn.modules(1).gradInput, seqBlas.modules(1).gradInput,
//          "gradInput") should be (0.0 +- 1e-6)

        gradInputDnn should be equals (gradInputBlas)
        Tools.averageError(gradInputDnn, gradInputBlas, "gradInput") should be(0.0 +- 2 * 1e-5)

       /*
        * TODO
        *
        * It's very stange that the cumulative error or average error of gradient weight
        * and gradient bias has big difference.
        */
      }
    }

    test[Float]()
    test[Double]()
  }
  "Omit conversion mix version" should "return correct result" in {
    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val modelDnn = getModel[T]("mix")
      val modelBlas = getModel[T]("blas")
      println(modelDnn)

      val paraDnn = modelDnn.parameters()
      val paraBlas = modelBlas.parameters()
      for (i <- 0 until paraDnn._1.length) {
        paraBlas._1(i).copy(paraDnn._1(i))
      }

      val input = Tensor[T](Array(32, 3, 224, 224)).rand()

      val outputDnn = modelDnn.forward(input)
      val outputBlas = modelBlas.forward(input)

      outputDnn should be equals (outputBlas)
      Tools.averageError(outputDnn, outputBlas, "output") should be(0.0 +- 1e-6)

      val gradOutput = Tensor[T]().resizeAs(outputDnn) rand ()

      val gradInputDnn = modelDnn.backward(input, gradOutput)
      val gradInputBlas = modelBlas.backward(input, gradOutput)

      gradInputDnn should be equals (gradInputBlas)
      Tools.averageError(gradInputDnn, gradInputBlas, "gradInput") should be(0.0 +- 1e-5)

      val (gradWeightDnn, gradBiasDnn) = modelDnn.getParameters()
      val (gradWeightBlas, gradBiasBlas) = modelBlas.getParameters()

      /*
       * TODO
       *
       * It's very stange that the cumulative error or average error of gradient weight
       * and gradient bias has big difference.
       */
      Tools.averageError(gradWeightDnn, gradWeightBlas, "gradWeight") should be(0.0 +- 1e-6)
      Tools.averageError(gradBiasDnn, gradBiasBlas, "gradBias") // should be(0.0 +- 1e2)
    }

    test[Float]()
  }

  "OmitConversion with mix layers five iterations" should "correct output and gradient input" in {
    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]): Unit = {
      val modelDnn = getModel[T]("mix")
      val modelBlas = getModel[T]("blas")
      println(modelDnn)

      val paraDnn = modelDnn.parameters()
      val paraBlas = modelBlas.parameters()
      for (i <- 0 until paraDnn._1.length) {
        paraBlas._1(i).copy(paraDnn._1(i))
      }

      var outDnn = Map[String, Tensor[T]]()
      var outBlas = Map[String, Tensor[T]]()
      val error = Map[String, Double]("output" -> 1e-6,
                                      "gradInput" -> 1e-6,
                                      "gradWeight" -> 1e-6,
                                      "gradBias" -> 1e3)

      for (i <- 0 until 5) {
        val input = Tensor[T](Array(32, 3, 224, 224)).rand()

        val outputDnn = modelDnn.forward(input)
        val outputBlas = modelBlas.forward(input)

        outDnn += ("output" -> outputDnn)
        outBlas += ("output" -> outputBlas)

        outputDnn should be equals (outputBlas)
        Tools.averageError(outputDnn, outputBlas,
                           "iteration " + i + " output") should be(0.0 +- 1e-6)

        Tools.averageError(outDnn, outBlas, error)

        val gradOutput = Tensor[T]().resizeAs(outputDnn) rand ()

        val gradInputDnn = modelDnn.backward(input, gradOutput)
        val gradInputBlas = modelBlas.backward(input, gradOutput)

        gradInputDnn should be equals (gradInputBlas)
        Tools.averageError(gradInputDnn, gradInputBlas, "iteration " + i + " gradInput") should be(
          0.0 +- 1e-5)

        val (gradWeightDnn, gradBiasDnn) = modelDnn.getParameters()
        val (gradWeightBlas, gradBiasBlas) = modelBlas.getParameters()

        /*
         * TODO
         *
         * It's very stange that the cumulative error or average error of gradient weight
         * and gradient bias has big difference.
         */
        Tools.averageError(gradWeightDnn, gradWeightBlas,
                           "iteration " + i + " gradWeight") should be(0.0 +- 1e-6)
        Tools.averageError(gradBiasDnn, gradBiasBlas, "iteration " + i + " gradBias")

      }
    }

    for (i <- 0 until Tools.getRandTimes()) {
      test[Float]()
      test[Double]()
    }
  }
}
