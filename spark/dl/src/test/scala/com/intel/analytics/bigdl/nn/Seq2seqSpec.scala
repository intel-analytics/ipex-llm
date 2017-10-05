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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.Seq2seq
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.torch.TH
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._

@com.intel.analytics.bigdl.tags.Parallel
class Seq2seqSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "A Seq2seq" should "work " in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 7
    val inputSize = 7
    val kernalW = 3
    val kernalH = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, inputSize, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize, 5, 5).rand

    val encoderCells = Array(ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val decoderCells = Array(ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val model = Seq2seq(encoderCells, decoderCells, feedPreviousOutput = true,
      outputLength = seqLength, decoderInputType = DecoderInputType.ENCODERINPUTLASTTIME)

    for (i <- 0 until 3) {
      model.forward(input).toTensor
      model.backward(input, gradOutput)
    }

    val model2 = Seq2seq(encoderCells, decoderCells,
      decoderInputType = DecoderInputType.ENCODERINPUTSPLIT)

    for (i <- 0 until 3) {
      val output = model2.forward(T(input, input)).toTensor
      model2.backward(input, gradOutput)
    }

    val model3 = Seq2seq(encoderCells, decoderCells,
      decoderInputType = DecoderInputType.ZEROS)

    for (i <- 0 until 3) {
      val output = model3.forward(input).toTensor
      model3.backward(input, gradOutput)
    }
  }

  "A Seq2seq" should "work2 " in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 2
    val inputSize = 5
    val seqLength = 3
    val seed = 123
    val batchSize = 2

    RNG.setSeed(seed)
    val encInSeq = Array(1.0, 2, 3, 3, 2, 1)
    val decInSeq = Array(1.0, 2, 3, 4, 4, 3, 2, 1)
    val decOutData = Array(2.0, 3, 4, 1, 1, 2, 3, 4)
    val input1 = Tensor[Double](encInSeq, Array(2, 3))
    val input = T(input1, Tensor[Double](decInSeq, Array(2, 4)))
    val decOutSeq = Tensor[Double](decOutData, Array(2, 4))
    
    val lstm = LSTM[Double](
      hiddenSize,
      hiddenSize)
    val encoderCells = Array(lstm).asInstanceOf[Array[Cell[Double]]]
    val enclstm = Recurrent().add(lstm).getParameters()._1

    val lstm2 = LSTM[Double](
      hiddenSize,
      hiddenSize)
    val decoderCells = Array(lstm2).asInstanceOf[Array[Cell[Double]]]
    val lstmP2 = lstm2.getParameters()._1

    val lookuptable1 = LookupTable(inputSize, hiddenSize)
    val lookuptableP1 = lookuptable1.getParameters()._1
    
    val code =
      s"""
         |
      |-- 1.4. Combine 1.1 and 1.3 to produce final model
         |require 'rnn'
         |
         |local enc1 = nn.Sequential()
         |lookup = nn.LookupTable(5, 2)
         |local parameters, gradParameters = lookup:getParameters()
         |parameters:copy(lookuptableP1)
         |enc1:add(lookup)
         |enc1:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
         |enc1output = enc1:forward(input)
         |
         |local enc = nn.Sequential()
         |local encLSTM = nn.FastLSTM(2, 2)         
         |enc:add(nn.Sequencer(encLSTM))
         |enc:add(nn.SelectTable(-1))
         |local parameters2, gradParameters2 = enc:getParameters()
         |--parameters2:copy(enclstm)         
         |         
         |local output = enc:forward(enc1output)
         |--local gradInput = model:backward(input, gradOutput)
    """.stripMargin
    scala.Seq

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input1, "lookuptableP1" -> lookuptableP1, "enclstm" -> enclstm),
      Array("output", "enc1output", "parameters2"))

    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val enc1output = torchResult("enc1output").asInstanceOf[Table]
    val luagradParameters = torchResult("parameters2").asInstanceOf[Tensor[Double]]

    val tt2 = lookuptable1.forward(input1)
    val tt3 = Recurrent().add(encoderCells.head).forward(tt2)

    val preEncoder = Sequential().add(lookuptable1)

    val lookuptable2 = LookupTable(inputSize, hiddenSize)
    val lookuptableP2 = lookuptable2.getParameters()._1

    val preDecoder = Sequential().add(lookuptable2)
    val seq2seq = Seq2seq(encoderCells, decoderCells, preEncoder = preEncoder,
      preDecoder = preDecoder, feedPreviousOutput = false,
      decoderInputType = DecoderInputType.ENCODERINPUTSPLIT)
    val model = Sequential().add(seq2seq)
    
    val linear = Linear(hiddenSize, inputSize)
    model.add(TimeDistributed(linear))
      .add(TimeDistributed(LogSoftMax()))

    val criterion = TimeDistributedCriterion(ClassNLLCriterion())

    
    val output = model.forward(input).toTensor
    val loss = criterion.forward(output, decOutSeq)
    
//    val gEdec = criterion.backward(output, decOutSeq)
//    model.backward(input, gEdec)
    val t = 0
  }
}

