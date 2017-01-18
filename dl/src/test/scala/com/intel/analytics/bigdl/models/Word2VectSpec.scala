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
package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.nn._
 import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adagrad, Optimizer, SGD}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.torch.TH
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.immutable
import scala.math._
import scala.util.Random

class Word2VectSpec extends FlatSpec with BeforeAndAfter with Matchers {
//  "aaa" should "aaa" in {
//    object ToSample {
//      def apply(nRows: Int, nCols: Int)
//      : ToSample =
//        new ToSample(nRows, nCols)
//    }
//
//    class ToSample(nRows: Int, nCols: Int)
//      extends Transformer[(Array[Double], Double), Sample[Double]] {
//      private val buffer = new Sample[Double]()
//      private var featureBuffer: Array[Double] = null
//      private var labelBuffer: Array[Double] = null
//
//      override def apply(prev: Iterator[(Array[Double], Double)]): Iterator[Sample[Double]] = {
//
//        prev.map(x => {
//
//          if (featureBuffer == null || featureBuffer.length < nRows * nCols) {
//            featureBuffer = new Array[Double](nRows * nCols)
//          }
//          if (labelBuffer == null || labelBuffer.length < nRows) {
//            labelBuffer = new Array[Double](nRows)
//          }
//
//          var i = 0
//          while (i < nRows) {
//            Array.copy(x._1, 0, featureBuffer, i * nCols, nCols)
//            labelBuffer(i) = x._2
//            i += 1
//          }
//
//          buffer.copy(featureBuffer, labelBuffer,
//            Array(nRows, nCols), Array(nRows))
//        })
//      }
//    }
//
//    // make up some data
//    val data = (0 to 100).collect {
//      case i if i > 75 || i < 25 ⇒
//        (0 to 100).collect {
//          case j if j > 75 || j < 25 ⇒
//            val res =
//              if (i > 75 && j < 25) 23.0
//              else if (i < 25 && j > 75) -45
//              else 0
//            (Array(i / 100.0 + 1, j / 100.0 + 2), res)
//        }
//    }.flatten
//
//
//    /**
//     * Batching samples into mini-batch
//     * @param batchSize The desired mini-batch size.
//     * @param sampleShape Shape of the training sample
//     */
//    case class OOMBatching(batchSize: Int, sampleShape: Array[Int])
//      extends Transformer[(Array[Double], Double), MiniBatch[Double]] {
//      override def apply(prev: Iterator[(Array[Double], Double)]): Iterator[MiniBatch[Double]] = {
//        new Iterator[MiniBatch[Double]] {
//          private val featureTensor: Tensor[Double] = Tensor[Double]()
//          private val labelTensor: Tensor[Double] = Tensor[Double]()
//          private var featureData: Array[Double] = null
//          private var labelData: Array[Double] = null
//          private val featureLength = sampleShape.product
//          private val labelLength = 1
//
//          override def hasNext: Boolean = prev.hasNext
//
//          override def next(): MiniBatch[Double] = {
//            if (prev.hasNext) {
//              var i = 0
//              while (i < batchSize && prev.hasNext) {
//                val sample = prev.next()
//                if (featureData == null || featureData.length < batchSize * featureLength) {
//                  featureData = new Array[Double](batchSize * featureLength)
//                }
//                if (labelData == null || labelData.length < batchSize * labelLength) {
//                  labelData = new Array[Double](batchSize * labelLength)
//                }
//                Array.copy(sample._1, 0, featureData, i * featureLength, featureLength)
//                labelData(i) = sample._2
//                i += 1
//              }
//              featureTensor
//                .set(Storage[Double](featureData), storageOffset = 1,
//                  sizes = Array(i) ++ sampleShape)
//              labelTensor.set(Storage[Double](labelData), storageOffset = 1, sizes = Array(i, 1))
//              MiniBatch(featureTensor, labelTensor)
//            }
//            else {
//              null
//            }
//          }
//        }
//      }
//    }
//
//    val numExecutors = 1
//    val numCores = 4
//
//    val sc = new SparkContext(
//      Engine.init(numExecutors, numCores, true).get
//        .setAppName("Sample_NN")
//        .set("spark.akka.frameSize", 64.toString)
//        .set("spark.task.maxFailures", "1")
//        .setMaster("local[4]")
//    )
//
//    import com.intel.analytics.bigdl.numeric.NumericDouble
//    val batchSize = 12
//
//    val dimInput = 2
//    val nHidden = 5
//    val sampleShape = Array(1, dimInput)
//    val batching = OOMBatching(batchSize, sampleShape)
//    val trainSetRDD = sc.makeRDD(data)
// //      .coalesce(numExecutors * numCores, true)
// //      .coalesce(numExecutors)
// //    val trainSet = DataSet.rdd(trainSetRDD) -> batching
//    val trainSet = DataSet.rdd(trainSetRDD) -> ToSample(1, dimInput) -> SampleToBatch(batchSize)
// // val trainSet =
  // DataSet.array(data.toArray) -> ToSample(1, dimInput) -> SampleToBatch(batchSize)
//
//    val layer1 = Linear[Double](dimInput, nHidden)
//    val layer2 = ReLU[Double]()
//    val layer3 = Linear[Double](nHidden, nHidden)
//    val layer4 = ReLU[Double]()
//    val output = Linear[Double](nHidden, 1) // Sum[Double](nInputDims = 1)
//
//    val model = Sequential[Double]()
//      .add(Reshape(Array(dimInput)))
//      .add(layer1)
//      .add(layer2)
//      .add(layer3)
//      .add(layer4)
//      .add(output)
//
//
//    val state =
//      T(
//        "learningRate" -> 0.01,
//        "weightDecay" -> 0.0005,
//        "momentum" -> 0.9,
//        "dampening" -> 0.0
//      )
//
//    val optimizer = Optimizer(
//      model = model,
//      dataset = trainSet,
//      criterion = new MSECriterion[Double]()
//    )
//
//    optimizer.
//      setState(state).
//      // setValidation(Trigger.everyEpoch, validationSet, Array(new Loss[Double])).
//      setOptimMethod(new Adagrad[Double]()).
//      optimize()
//    print(model.getParameters())
//  }

  "Word2Vec Float" should "generate correct output" in {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }

    val seed = 10
    RNG.setSeed(seed)
    val vocabSize = 10
    val embeddingSize = 10
    val numNegSamples = 5
    val batchSize = 2

    def getModel: Module[Float] = {
      val wordVectors = LookupTable(vocabSize, embeddingSize)
 //      wordVectors.weight.apply1(x => 1)
      new Sequential()
        .add(wordVectors)
         .add(ConcatTable()
           .add(Narrow(2, 1, 1))
           .add(Narrow(2, 2, numNegSamples + 1)))
         .add(MM(transA = false, transB = true))
        .add(Sigmoid())
    }

    val input = Tensor(batchSize, 2 + numNegSamples).apply1( e => Random.nextInt(vocabSize) + 1)
//    val labels = Tensor(batchSize, 2 + numNegSamples, embeddingSize).zero()
    val labels = Tensor(batchSize, 1 + numNegSamples).zero()

//    for (i <- 1 to batchSize) {
//      for (j <- 1 to 2 + numNegSamples) {
//        labels.setValue(i, j, 1, 1)
//      }
//    }

     for (i <- 1 to batchSize) {
       labels.setValue(i, 1, 1)
     }

    val model = getModel


    val code =
      "torch.setdefaulttensortype('torch.FloatTensor')" +
        "torch.manualSeed(" + seed + ")\n" +
        "word_vecs = nn.LookupTable(" + vocabSize + ", " +embeddingSize + ")" +
      s"""
        |   -- word_vecs.weight:apply(function() return 1 end)
        |   w2v = nn.Sequential()
        |   w2v:add(word_vecs)
        |   w2v:add(nn.ConcatTable()
        |                   :add(nn.Narrow(2,1,1))
        |                   :add(nn.Narrow(2,2, ${numNegSamples + 1})))
        |    w2v:add(nn.MM(false, true)) -- dot prod and sigmoid to get probabilities
        |    w2v:add(nn.Sigmoid())
        |
        |    model = w2v
        |
        |     local parameters, gradParameters = model:getParameters()
        |                parameters_initial = parameters : clone()
        |                gradParameters_initial = gradParameters : clone()
        |
        |                local criterion = nn.BCECriterion()
        |                state = {
        |                  learningRate = 1e-2,
        |                }
        |
        |         feval = function(x)
        |              model:forward(input)
        |              criterion:forward(model.output, labels)
        |              model:zeroGradParameters()
        |              criterion:backward(model.output, labels)
        |              model:backward(input, criterion.gradInput)
        |              return criterion.output, gradParameters
        |           end
        |
        |             for i = 1, 5, 1 do
        |              w, err = optim.sgd(feval, parameters, state)
        |             end
        | --print(model.modules[2].output)
        |                output=model.output
        |                gradOutput=criterion.gradInput
        |                err = criterion.output
        |                gradInput = model.gradInput
      """.stripMargin

    TH.runNM(code, immutable.Map(
      "input" -> input,
      "labels" -> labels),
      Array("output", "gradOutput", "err", "parameters_initial",
        "gradParameters_initial", "gradInput", "model"))

    val parameterTorch = TH.map("parameters_initial").asInstanceOf[Tensor[Float]]
    val parameters = model.getParameters()._1

    for (i <- 0 until parameters.nElement()) {
      if (abs(parameters.storage().array()(i) - parameterTorch.storage().array()(i)) > 1e-8) {
        println(s"${parameters.storage().array()(i)} ${parameterTorch.storage().array()(i)}")
      }
    }

    val (weights, grad) = model.getParameters()
    val criterion = BCECriterion[Float]()

    val state = T("learningRate" -> 1e-2)

    val sgd = new SGD[Float]

    def feval(x: Tensor[Float]): (Float, Tensor[Float]) = {
      model.forward(input)
      criterion.forward(model.output.asInstanceOf[Tensor[Float]], labels)
      model.zeroGradParameters()
      val gradOutputTest = criterion.backward(model.output.asInstanceOf[Tensor[Float]], labels)
      model.backward(input, gradOutputTest)
      (criterion.output, grad)
    }
    for (i <- 1 to 5) {
      sgd.optimize(feval, weights, state)
    }

    val output = TH.map("output").asInstanceOf[Tensor[Float]]
    val outputTest = model.output.toTensor[Float]
    var abss = 0.0
    for (i <- 0 until outputTest.nElement()) {
      val tmp = abs(outputTest.storage().array()(i) - output.storage().array()(i))
      abss += tmp
    }
    println(s"outputAbs:$abss")
    output should be (outputTest)
    assert(abss < 1e-2)


    val errTest = criterion.output
    val err = TH.map("err").asInstanceOf[Double]
    println(s"${abs(errTest - err)}")
//    assert(abs(errTest - err) < 1.5e-6)

    val gradOutputTest = criterion.backward(outputTest, labels)
    val gradOutput = TH.map("gradOutput").asInstanceOf[Tensor[Float]]
    abss = 0.0
    for (i <- 0 until gradOutputTest.nElement()) {
      val tmp = abs(gradOutputTest.storage().array()(i) - gradOutput.storage().array()(i))
      abss += tmp
    }

    gradOutput should be (gradOutputTest)
    assert(abss < 2e-6)
    println(s"gradOutputTestAbs:$abss")

    val gradInput = model.gradInput.asInstanceOf[Tensor[Float]]
    val gradInputTorch = TH.map("gradInput").asInstanceOf[Tensor[Float]]

    abss = 0.0
    for (i <- 0 until gradInputTorch.nElement()) {
      val tmp = abs(gradInputTorch.storage().array()(i) - gradInput.storage().array()(i))
      abss += tmp
    }
    println(s"gradInputTestAbs:$abss")
  }
}
