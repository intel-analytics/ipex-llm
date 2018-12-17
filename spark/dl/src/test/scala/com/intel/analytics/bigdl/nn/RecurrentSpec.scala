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

import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.math._
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class RecurrentSpec extends FlatSpec with Matchers {

  "Recurrent" should "outputs correct hiddens" in {
    val hiddenSize = 4
    val batchSize = 3
    val inputSize = 6
    val seqLength = 5
    val seed = 100

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(batchSize, seqLength, inputSize)).rand()
    input.select(1, 2).zero()

    val rec = Recurrent[Double](maskZero = true)
    val initHidden = T(
      Tensor[Double](Array(batchSize, hiddenSize)).rand(),
      Tensor[Double](Array(batchSize, hiddenSize)).rand()
    )
    rec.setHiddenState(initHidden)

    val lstm = LSTM[Double](inputSize, hiddenSize)
    val model = Sequential[Double]()
      .add(rec
        .add(lstm))

    model.forward(input)

    lstm.output.toTable[Table](2).toTable[Tensor[Double]](1)
      .select(1, 2) should be (initHidden[Tensor[Double]](1).select(1, 2))
  }

  "Recurrent" should "ouputs correclty" in {
    val hiddenSize = 4
    val batchSize = 3
    val inputSize = 6
    val seqLength = 5
    val seed = 100

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(batchSize, seqLength, inputSize)).rand()
    input.select(1, 2).select(1, seqLength).zero()

    val rec = Recurrent[Double](maskZero = true)

    val model = Sequential[Double]()
      .add(rec
        .add(LSTM[Double](inputSize, hiddenSize)))

    val output = model.forward(input)

    output.toTensor[Double].select(1, 2).select(1, seqLength).abs().max() should be (0)
  }

  "A Recurrent" should "call getTimes correctly" in {
    val hiddenSize = 128
    val inputSize = 1280
    val outputSize = 128
    val time = 30
    val batchSize1 = 100
    val batchSize2 = 8
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(LSTM[Double](inputSize, hiddenSize)))
      .add(Select(2, 1))
    //      .add(Linear[Double](hiddenSize, outputSize))

    val input = Tensor[Double](Array(batchSize1, time, inputSize)).rand
    val gradOutput = Tensor[Double](batchSize1, outputSize).rand

    model.clearState()

    for (i <- 1 to 10) {
      model.forward(input)
      model.backward(input, gradOutput)
    }
    model.resetTimes()

    var st = System.nanoTime()
    model.forward(input)
    val etaForward = System.nanoTime() - st
    println(s"forward eta = ${etaForward}")
    st = System.nanoTime()
    model.backward(input, gradOutput)
    val etaBackward = System.nanoTime() - st
    println(s"backward eta = ${etaBackward}")
    println()
    var forwardSum = 0L
    var backwardSum = 0L

    model.getTimes.foreach(x => {
      println(x._1 + ", " + x._2 + ", " + x._3)
      forwardSum += x._2
      backwardSum += x._3
    })
    println(s"forwardSum = ${forwardSum}")
    println(s"backwardSum = ${backwardSum}")

    assert(abs((etaForward - forwardSum) / etaForward) < 0.01)
    assert(abs((etaBackward - backwardSum) / etaBackward) < 0.01)

    val times = model.getTimesGroupByModuleType()
    times.length should be (6)
    times.map(_._2).sum should be (etaForward +- etaForward / 100)
    times.map(_._3).sum should be (etaBackward +- etaBackward / 100)
  }

  "A Recurrent with LSTMPeephole cell" should "add batchNormalization correctly" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val batchSize = 4
    val time = 2
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double](BatchNormParams())
        .add(LSTMPeephole[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    println(model)

    val input = Tensor[Double](batchSize, time, inputSize)
    val gradOutput = Tensor[Double](batchSize, time, outputSize)

    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)

    println("add normalization")
  }

  "A Recurrent with GRU cell" should "add batchNormalization correctly" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val batchSize = 4
    val time = 2
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double](BatchNormParams())
        .add(GRU[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    println(model)

    val input = Tensor[Double](batchSize, time, inputSize)
    val gradOutput = Tensor[Double](batchSize, time, outputSize)

    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)

    println("add normalization")
  }

  "A Recurrent with LSTM cell" should "add batchNormalization correctly" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val batchSize = 4
    val time = 2
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double](BatchNormParams())
        .add(LSTM[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    println(model)

    val input = Tensor[Double](batchSize, time, inputSize)
    val gradOutput = Tensor[Double](batchSize, time, outputSize)

    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)

    println("add normalization")
  }

  "A Recurrent with SimpleRNN cell" should "add batchNormalization correctly" in {
    val hiddenSize = 4
    val inputSize = 5
    val batchSize = 2
    val time = 2
    val seed = 100
    RNG.setSeed(seed)

    val cell = RnnCell[Double](inputSize, hiddenSize, ReLU[Double]())
    val model = Sequential[Double]()
      .add(Recurrent[Double](BatchNormParams())
        .add(cell))

    val (weightsArray, gradWeightsArray) = model.parameters()
    weightsArray(0).set(Tensor[Double](Array(0.038822557355026155,
      0.15308625574211315, -0.1982324504512677, -0.07866809418407278,
      -0.06751351799422134, 0.023597193777786962, 0.3083771498964048,
      -0.31429738377130323, -0.4429929170091549, -0.30704694098520874,
      -0.33847886911170505, -0.2804322767460886, 0.15272262323432112,
      -0.2592875227066882, 0.2914515643266326, -0.0422707164265147,
      -0.32493950675524846, 0.3310656372548169, 0.06716552027607742, -0.39025554201755425),
      Array(4, 5)))
    weightsArray(1).set(Tensor[Double](Array(0.3500089930447004,
      0.11118793394460541,
      -0.2600975267200473,
      0.020882861472978465), Array(4)))

    weightsArray(2).set(Tensor[Double](Array(0.18532821908593178,
    0.5622962701600045,
    0.10837689251638949,
    0.005817196564748883),
      Array(4)))

    weightsArray(4).set(Tensor[Double](Array(-0.28030250454321504,
      -0.19257679535076022, 0.4786237839143723, 0.45018431078642607,
    0.31168314907699823, -0.37334575527347624, -0.3280589876230806, -0.4210121303331107,
    0.31622475129552186, -0.18864686344750226, -0.22592625673860312, 0.13238358590751886,
    -0.06829581526108086, 0.1993589240591973, 0.44002981553785503, 0.14196494384668767),
      Array(4, 4)))
    weightsArray(5).set(Tensor[Double](Array(0.3176493758801371,
    0.4200237800832838,
    -0.16388805187307298,
    -0.20112364063970745), Array(4)))

    val input = Tensor[Double](Array(0.1754104527644813, 0.5687455364968628,
      0.3728320465888828, 0.17862433078698814, 0.005688507109880447,
    0.5325737004168332, 0.2524263544473797, 0.6466914659831673, 0.7956625143997371,
      0.14206538046710193, 0.015254967380315065, 0.5813889650162309, 0.5988433782476932,
      0.4791899386327714, 0.6038045417517424, 0.3864191132597625, 0.1051476860884577,
      0.44046495063230395, 0.3819434456527233, 0.40475733182393014),
      Array(batchSize, time, inputSize))
    val gradOutput = Tensor[Double](Array(0.015209059891239357, 0.08723655440707856,
      1.2730716350771312, 0.17783007683002253, 0.9809208554215729,
      0.7760053128004074, 0.05994199030101299, 0.550958373118192,
    1.1344734990039382, -0.1642483852831349, 0.585060822398516, 0.6124844773937481,
    0.7424796849954873, 0.95687689865008, 0.6301839421503246, 0.17582130827941),
      Array(batchSize, time, hiddenSize))

    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)

    output should be (Tensor[Double](Array(0.6299258968399799,
      1.3642297404555106, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0361454784986459,
    0.25696697120146866, 0.19273657953649884, 0.0, 0.0,
    0.12799813658263415, 0.24882574216093045, 0.0, 0.0),
      Array(batchSize, time, hiddenSize)))

    gradInput should be (Tensor[Double](Array(-0.027275798942804907,
      -0.1383686829541865, 0.16717516624801407, 0.11372249422239256,
      0.08955797331273728, -0.08873463347124873, -0.38487333246986216,
      0.48437215964262187, 0.2417856458208813, 0.19281229062491595,
    0.03225607513585589, -0.36656214055421815, 0.2795038794253451,
      0.8385161794048844, 0.549019363159085, 0.08375435727819777,
      0.8898041559782669, -0.9310512053159811, -1.1940243194481583, -0.8313896270967381),
      Array(batchSize, time, inputSize)))
  }

  "A Recurrent" should "converge when batchSize changes" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val time = 4
    val batchSize1 = 5
    val batchSize2 = 8
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh[Double]())))
      .add(Select(2, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    val input1 = Tensor[Double](Array(batchSize1, time, inputSize)).rand
    val input2 = Tensor[Double](batchSize2, time, inputSize).rand

    val gradOutput1 = Tensor[Double](batchSize1, outputSize).rand
    val gradOutput2 = Tensor[Double](batchSize2, outputSize).rand

    model.clearState()

    model.forward(input1)
    model.backward(input1, gradOutput1)
    val gradInput1 =
      Tensor[Double](batchSize1, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output1 = Tensor[Double](batchSize1, outputSize).copy(model.output.toTensor[Double])

    model.clearState()

    model.forward(input2)
    model.backward(input2, gradOutput2)
    val gradInput2 =
      Tensor[Double](batchSize2, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output2 = Tensor[Double](batchSize2, outputSize).copy(model.output.toTensor[Double])

    model.forward(input1)
    model.backward(input1, gradOutput1)
    val gradInput1compare =
      Tensor[Double](batchSize1, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output1compare = Tensor[Double](batchSize1, outputSize)
      .copy(model.output.toTensor[Double])

    model.forward(input2)
    model.backward(input2, gradOutput2)
    val gradInput2compare =
      Tensor[Double](batchSize2, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output2compare = Tensor[Double](batchSize2, outputSize)
      .copy(model.output.toTensor[Double])

    model.hashCode()

    output1 should be (output1compare)
    output2 should be (output2compare)

    gradInput1 should be (gradInput1compare)
    gradInput2 should be (gradInput2compare)
  }

  "A Recurrent Language Model Module" should "converge" in {

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh[Double]())))
      .add(Select(1, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val criterion = CrossEntropyCriterion[Double]()
    val logSoftMax = LogSoftMax[Double]()

    val (weights, grad) = model.getParameters()

    val input = Tensor[Double](Array(1, 5, inputSize))
    val labels = Tensor[Double](Array(1, 5))
    for (i <- 1 to 5) {
      val rdmLabel = Math.ceil(RNG.uniform(0.0, 1.0)*inputSize).toInt
      val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0)*inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    for (i <- 1 to 50) {
      val (_, loss) = sgd.optimize(feval, weights, state)
      println(s"${i}-th loss = ${loss(0)}")
    }

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(2)._2

    labels.squeeze() should be (prediction.squeeze())
  }

  "A Recurrent Module" should "converge in batch mode" in {

    val batchSize = 10
    val nWords = 5
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh())))
      .add(Select(2, nWords))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val criterion = CrossEntropyCriterion[Double]()
    val logSoftMax = LogSoftMax[Double]()

    val (weights, grad) = model.getParameters()
    val data = Array(-0.051649563734092574, 0.3491947190721401, -0.42027052029370376,
      0.4301486898079941, 0.2468666566291215, -0.39359984949207866, 0.045578554526030046,
      0.3493149141337017, -0.1063711823523733, 0.06878279210527599, 0.02617610773350143,
      0.21688042352505815, 0.4086431210923443, 0.1164400576908104, -0.289954236617675,
      0.07320188583739445, -0.34140032046902746, -0.42893228205681105, 0.3246284763380037,
      -0.259360108472857, -0.3802506202721077, 0.039967368527818625, 0.2907736835216905,
      0.24070392389100653, 0.04340493865311146, 0.17115563713014126, -0.22163061727769673,
      -0.08795360312797129, -0.07548240781761706, 0.02638246468268335, 0.34477613493800163,
      -0.35139515763148665, -0.4952811379916966, -0.3432889161631465, -0.3784308801405132,
      -0.31353281694464386, 0.17074908362701535, -0.2898922632448375, 0.32585275499150157,
      -0.047260097693651915, -0.36329341283999383, 0.3701426349580288, 0.07509333454072475,
      -0.43631896027363837, 0.3361318111419678, -0.24930476839654148, -0.4246050880756229,
      -0.21410430688410997, -0.4885992160998285, 0.352395088179037, -0.45157943526282907,
      0.47500649164430797, -0.142877290956676, 0.38485329202376306, 0.1656933748163283,
      -0.14049215079285204, -0.48861038917675614, 0.09885894856415689, -0.3920822301879525,
      -0.14520439435727894, 0.401013100752607, -0.15980978682637215, 0.2948787631466985,
      -0.3219190139789134, 0.31146098021417856, -0.2623057949822396, 0.14027805789373815,
      -0.45513772079721093, 0.1247795126400888)
    val tmp = Tensor[Double](data, Array(data.size, 1))
    weights.copy(tmp)

    val input = Tensor[Double](Array(batchSize, nWords, inputSize))
    val labels = Tensor[Double](batchSize)
    for (b <- 1 to batchSize) {
      for (i <- 1 to nWords) {
        val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0) * inputSize).toInt
        input.setValue(b, i, rdmInput, 1.0)
      }
      val rdmLabel = Math.ceil(RNG.uniform(0.0, 1.0) * outputSize).toInt
      labels.setValue(b, rdmLabel)
    }

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    for (i <- 1 to 50) {
      val (_, loss) = sgd.optimize(feval, weights, state)
      println(s"${i}-th loss = ${loss(0)}")
    }

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(2)._2

    labels.squeeze() should be (prediction.squeeze())
  }

  "A Recurrent Module" should "perform correct gradient check" in {

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 10
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh())))
      .add(Select(1, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val input = Tensor[Double](Array(1, 5, inputSize))
    val labels = Tensor[Double](Array(1, 5))
    for (i <- 1 to 5) {
      val rdmLabel = Math.ceil(Math.random()*inputSize).toInt
      val rdmInput = Math.ceil(Math.random()*inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    println("gradient check for input")
    val gradCheckerInput = new GradientChecker(1e-2, 1)
    val checkFlagInput = gradCheckerInput.checkLayer[Double](model, input)
    println("gradient check for weights")
    val gradCheck = new GradientCheckerRNN(1e-2, 1)
    val checkFlag = gradCheck.checkLayer(model, input, labels)
  }

  "Recurrent dropout" should "work correclty" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seqLength = 5
    val seed = 1

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(1, seqLength, inputSize))
    for (i <- 1 to seqLength) {
      val rdmInput = 3
      input.setValue(1, i, rdmInput, 1.0)
    }

    println(input)
    val gru = GRU[Double](inputSize, hiddenSize, 0.2)
    val model = Recurrent[Double]().add(gru)

    val field = model.getClass.getDeclaredField("cells")
    field.setAccessible(true)
    val cells = field.get(model).asInstanceOf[ArrayBuffer[Cell[Double]]]

    val dropoutsRecurrent = model.asInstanceOf[Container[_, _, Double]].findModules("Dropout")
    val dropoutsCell = gru.cell.asInstanceOf[Container[_, _, Double]].findModules("Dropout")
    val dropouts = dropoutsRecurrent ++ dropoutsCell
    dropouts.size should be (6)

    val output = model.forward(input)
    val noises1 = dropouts.map(d => d.asInstanceOf[Dropout[Double]].noise.clone())
    noises1(0) should not be noises1(1)

    val noises = dropoutsCell.map(d => d.asInstanceOf[Dropout[Double]].noise.clone())
    for (i <- dropoutsCell.indices) {
      cells.foreach(c => {
        val noise = c.cell.asInstanceOf[Container[_, _, Double]]
          .findModules("Dropout")(i)
          .asInstanceOf[Dropout[Double]]
          .noise
        noise should be(noises(i))
      })
    }


    model.forward(input)

    var flag = true
    for (i <- dropoutsCell.indices) {
      cells.foreach(c => {
        val newNoises = c.cell.asInstanceOf[Container[_, _, Double]]
          .findModules("Dropout")
        val noise = newNoises(i).asInstanceOf[Dropout[Double]].noise
        flag = flag && (noise == noises(i))
      })
    }

    flag should be (false)
  }

  "A Recurrent Module" should "work with get/set state " in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 10
    val seed = 100
    val batchSize = 1
    val time = 4
    RNG.setSeed(seed)

    val rec = Recurrent[Double]()
      .add(RnnCell[Double](inputSize, hiddenSize, Tanh()))
    val model = Sequential[Double]()
      .add(rec)

    val input = Tensor[Double](Array(batchSize, time, inputSize)).rand

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val state = rec.getHiddenState()

    state.toTensor[Double].map(output.asInstanceOf[Tensor[Double]].select(2, time), (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })

    rec.setHiddenState(state)
    model.forward(input)
  }

  "A Recurrent Module" should "work good with copy " in {
    val input = Tensor[Float](3, 2, 6, 10).randn()
    val input1 = input.select(2, 1).clone()
    val input2 = input.select(2, 2).clone()

    val arrInput = new ArrayBuffer[Tensor[Float]](2)
    arrInput.append(input1)
    arrInput.append(input2)

    val output1 = Tensor[Float]()
    val output2 = Tensor[Float]().resizeAs(input)

    Recurrent.selectCopy(input, 2, output1)
    output1 should be (input.select(2, 2))

    Recurrent.copy(arrInput, output2)
    output2 should be (input)
  }

  "A Recurrent Module" should "work after reset " in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh())))
      .add(Select(1, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    val input = Tensor[Double](Array(1, 5, inputSize))
    val output1 = model.forward(input).toTensor[Double].clone()
    model.reset()
    model.forward(input)
  }

}

class RecurrentSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val recurrent = Recurrent[Float]().setName("recurrent")
      .add(RnnCell[Float](5, 4, Tanh[Float]()))
    val input = Tensor[Float](Array(10, 5, 5)).apply1(_ => Random.nextFloat())
    runSerializationTest(recurrent, input)
  }
}
