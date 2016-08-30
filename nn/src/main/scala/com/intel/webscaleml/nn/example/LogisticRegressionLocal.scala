package com.intel.webscaleml.nn.example

import com.intel.webscaleml.nn.nn._
import com.intel.webscaleml.nn.optim.{LBFGS, SGD, Adagrad}
import com.intel.webscaleml.nn.tensor.{Storage, T, torch, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object LogisticRegressionLocal {
  var curIter = 1
  var correct = 0
  var count = 0
  var l = 0.0
  var wallClockTime = 0L

  def loadData(file : String): (Tensor[Double], Tensor[Double]) = {
    val inputs = torch.Tensor[Double](735, 2)
    val outputs = torch.Tensor[Double](735)

    var i = 1
    scala.io.Source.fromFile(file).getLines().foreach{line =>
      val record = line.split(",")
      val label = record(0).toDouble
      val inp = torch.Tensor[Double](torch.storage(record.slice(1, record.length).map(_.toDouble)))
      outputs(Array(i)) = label
      inputs(i).copy(inp)
      i += 1
    }
    (inputs, outputs)
  }

  def loadSparseData(file : String) : (Tensor[Double], Tensor[Double]) = {
    val inputRowIndex = new ArrayBuffer[Int]()
    val inputColIndex = new ArrayBuffer[Int]()
    val values = new ArrayBuffer[Double]()
    val inputShape = Array(735, 2)
    val outputIndices = new ArrayBuffer[Int]()
    val outputValues = new ArrayBuffer[Double]()
    val outputShape = Array(735)

    var i = 1
    scala.io.Source.fromFile(file).getLines().foreach{line =>
      val record = line.split(",")
      val label = record(0).toDouble
      val inp = record.slice(1, record.length).map(_.toDouble)
      if(inp(0) != 0){
        inputRowIndex.append(i)
        inputColIndex.append(1)
        values.append(inp(0))
      }
      inputRowIndex.append(i)
      inputColIndex.append(2)
      values.append(inp(1))

      if(label != 0){
        outputIndices.append(i)
        outputValues.append(label)
      }

      i += 1
    }

    val inputs = torch.Tensor[Double](Array(inputRowIndex.toArray, inputColIndex.toArray), torch.storage[Double](values.toArray), inputShape)
    val outputs = torch.Tensor[Double](Array(outputIndices.toArray), torch.storage[Double](outputValues.toArray), outputShape)

    (inputs, outputs)
  }

  def loadDatas(file : String, sparse : Boolean = false): (Tensor[Double], Tensor[Double]) = {
    val inputs = new ArrayBuffer[Tensor[Double]]
    val outputs = new ArrayBuffer[Double]

    var i = 1
    scala.io.Source.fromFile(file).getLines().foreach{line =>
      val record = line.split(",")
      val label = record(0).toDouble
      val inp = torch.Tensor[Double](torch.storage(record.slice(1, record.length).map(_.toDouble)))
//      val (label, inp) = parse(line, 15)

      outputs.append(label)
      inputs.append(inp)
      i += 1
    }

    if(sparse){
      (toSparseTensor(inputs.toArray), torch.Tensor[Double](torch.storage(outputs.toArray)))
    } else {
      (toTensor(inputs.toArray), torch.Tensor[Double](torch.storage(outputs.toArray)))
    }
  }

  def parse(l: String, modelSize: Int): (Int, Tensor[Double]) = {
    val originalModelSize = 14
    val tokens = l.split(", ")
    require(tokens.length == modelSize, s"invalid model size ${modelSize} total length|${tokens(0)}| ${tokens.length}")
    require((modelSize - 1) % originalModelSize == 0, s"invalid model size ${modelSize}")
    val values = new Array[Double](modelSize - 1)
    val repeat = modelSize / originalModelSize
    for (i <- 0 until repeat) {
      // age
      values(i * originalModelSize) = tokens(i * originalModelSize).toDouble

      // work class
      values(i * originalModelSize + 1) = tokens(i * originalModelSize + 1) match {
        case "?" => 0
        case "Private" => 1
        case "Self-emp-not-inc" => 2
        case "Self-emp-inc" => 3
        case "Federal-gov" => 4
        case "Local-gov" => 5
        case "State-gov" => 6
        case "Without-pay" => 7
        case "Never-worked" => 8
        case other => throw new IllegalArgumentException("Invalid workclass " + other)
      }

      //fnlwgt
      values(i * originalModelSize + 2) = tokens(i * originalModelSize + 2).toDouble

      // education
      values(i * originalModelSize + 3) = tokens(i * originalModelSize + 3) match {
        case "?" => 0
        case "Bachelors" => 1
        case "Some-college" => 2
        case "11th" => 3
        case "HS-grad" => 4
        case "Prof-school" => 5
        case "Assoc-acdm" => 6
        case "Assoc-voc" => 7
        case "9th" => 8
        case "7th-8th" => 9
        case "12th" => 10
        case "Masters" => 11
        case "1st-4th" => 12
        case "10th" => 13
        case "Doctorate" => 14
        case "5th-6th" => 15
        case "Preschool" => 16
        case other => throw new IllegalArgumentException("Invalid education " + other)
      }

      // education-num
      values(i * originalModelSize + 4) = tokens(i * originalModelSize + 4).toDouble

      // marital-status
      values(i * originalModelSize + 5) = tokens(i * originalModelSize + 5) match {
        case "?" => 0
        case "Married-civ-spouse" => 1
        case "Divorced" => 2
        case "Never-married" => 3
        case "Separated" => 4
        case "Widowed" => 5
        case "Married-spouse-absent" => 6
        case "Married-AF-spouse" => 7
        case other => throw new IllegalArgumentException("Invalid marital-status " + other)
      }

      // occupation
      values(i * originalModelSize + 6) = tokens(i * originalModelSize + 6) match {
        case "?" => 0
        case "Tech-support" => 1
        case "Craft-repair" => 2
        case "Other-service" => 3
        case "Sales" => 4
        case "Exec-managerial" => 5
        case "Prof-specialty" => 6
        case "Handlers-cleaners" => 7
        case "Machine-op-inspct" => 8
        case "Adm-clerical" => 9
        case "Farming-fishing" => 10
        case "Transport-moving" => 11
        case "Priv-house-serv" => 12
        case "Protective-serv" => 13
        case "Armed-Forces" => 14
        case other => throw new IllegalArgumentException("Invalid occupation " + other)
      }

      // relationship
      values(i * originalModelSize + 7) = tokens(i * originalModelSize + 7) match {
        case "?" => 0
        case "Wife" => 1
        case "Own-child" => 2
        case "Husband" => 3
        case "Not-in-family" => 4
        case "Other-relative" => 5
        case "Unmarried" => 6
        case other => throw new IllegalArgumentException("Invalid relationship " + other)
      }

      // race
      values(i * originalModelSize + 8) = tokens(i * originalModelSize + 8) match {
        case "?" => 0
        case "White" => 1
        case "Asian-Pac-Islander" => 2
        case "Amer-Indian-Eskimo" => 3
        case "Other" => 4
        case "Black" => 5
        case other => throw new IllegalArgumentException("Invalid race " + other)
      }

      // sex
      values(i * originalModelSize + 9) = tokens(i * originalModelSize + 9) match {
        case "?" => 0
        case "Female" => 1
        case "Male" => 2
        case other => throw new IllegalArgumentException("Invalid sex " + other)
      }

      // capital-gain
      values(i * originalModelSize + 10) = tokens(i * originalModelSize + 10).toDouble

      // capital-loss
      values(i * originalModelSize + 11) = tokens(i * originalModelSize + 11).toDouble

      // hours-per-week
      values(i * originalModelSize + 12) = tokens(i * originalModelSize + 12).toDouble

      // native-country
      values(i * originalModelSize + 13) = tokens(i * originalModelSize + 13) match {
        case "?" => 0
        case "United-States" => 1
        case "Cambodia" => 2
        case "England" => 3
        case "Puerto-Rico" => 4
        case "Canada" => 5
        case "Germany" => 6
        case "Outlying-US(Guam-USVI-etc)" => 7
        case "India" => 8
        case "Japan" => 9
        case "Greece" => 10
        case "South" => 11
        case "China" => 12
        case "Cuba" => 13
        case "Iran" => 14
        case "Honduras" => 15
        case "Philippines" => 16
        case "Italy" => 17
        case "Poland" => 18
        case "Jamaica" => 19
        case "Vietnam" => 20
        case "Mexico" => 21
        case "Portugal" => 22
        case "Ireland" => 23
        case "France" => 24
        case "Dominican-Republic" => 25
        case "Laos" => 26
        case "Ecuador" => 27
        case "Taiwan" => 28
        case "Haiti" => 29
        case "Columbia" => 30
        case "Hungary" => 31
        case "Guatemala" => 32
        case "Nicaragua" => 33
        case "Scotland" => 34
        case "Thailand" => 35
        case "Yugoslavia" => 36
        case "El-Salvador" => 37
        case "Trinadad&Tobago" => 38
        case "Peru" => 39
        case "Hong" => 40
        case "Holand-Netherlands" => 41
        case other => throw new IllegalArgumentException("Invalid native-country " + other)
      }
    }

    val tag = tokens(modelSize - 1) match {
      case "<=50K" => 0
      case ">50K" => 1
      case "<=50K." => 0
      case ">50K." => 1
      case other => throw new IllegalArgumentException("Invalid income tag " + other)
    }

    (tag, torch.Tensor[Double](torch.storage(values)))
  }

  def toTensor(data : Array[Tensor[Double]]) : Tensor[Double] = {
    val result = torch.Tensor[Double](data.length, data(0).size(1))

    var i = 0
    while(i < data.length){
      result(i + 1).copy(data(i))
      i += 1
    }
    result
  }

  def toSparseTensor(data : Array[Tensor[Double]]) : Tensor[Double] = {
    val shape = Array(data.length, data(0).size(1))
    val rowIndices = new ArrayBuffer[Int]()
    val colIndices = new ArrayBuffer[Int]()
    val values = new ArrayBuffer[Double]()

    var i = 0
    while(i < data.length){
      var j = 1
      while(j <= data(i).size(1)){
        if(data(i)(Array(j)) != 0) {
          rowIndices.append(i+1)
          colIndices.append(j)
          values.append(data(i)(Array(j)))
        }
        j += 1
      }
      i += 1
    }
    torch.Tensor[Double](Array(rowIndices.toArray, colIndices.toArray), torch.storage[Double](values.toArray), shape)
  }

  def getTestData(): Tensor[Double] = {
    val test = torch.Tensor[Double](45, 2)
    for(i <- 1 to 15){
      test(Array(i, 1)) = 1
      test(Array(i, 2)) = i + 23
      test(Array(i + 15, 1)) = 2
      test(Array(i + 15, 2)) = i + 23
      test(Array(i + 30, 1)) = 3
      test(Array(i + 30, 2)) = i + 23
    }
    test
  }

  def getModel(): Module[Double] = {
    val lr = new Sequential[Double]()
    lr.add(new Linear(2, 1))
    lr.add(new Sigmoid())
    lr
  }

  def main(args: Array[String]) {
//    val (trainData, trainLabel) = loadDatas("/home/xin/datasets/adult/adult.data")
//    val (trainData, trainLabel) = loadDatas("/home/xin/datasets/ds1.10/ds1.10.csv")
//    val (trainData, trainLabel) = loadData("algorithms/src/main/scala/com/intel/webscaleml/nn/example/example-logistic-regression.csv")
    val (trainData, trainLabel) = loadData("algorithms/src/main/scala/com/intel/webscaleml/nn/example/example.csv")
//    val (trainData2, trainLabel2) = loadDatas("algorithms/src/main/scala/com/intel/webscaleml/nn/example/example.csv")

    val testData = getTestData()

    val batchSize = 5
    val trainSize = trainData.size(1)
    val model = getModel()
//    val model = Cifar.getModel("/home/xin/IdeaProjects/demos/logistic-regression/lr.net")
//    val criterion = new ClassNLLCriterion()
    val criterion = new BCECriterion[Double]()

    val (masterWeights, masterGrad) = model.getParameters()
//    val optm = new torchLBFGS()
    val optm = new SGD[Double]()
    val config = T ("learningRate" -> 1e-2, "weightDecay" -> 0.0,
      "momentum" -> 0.0, "learningRateDecay" -> 1e-6)

    var epoch = 1

    while (epoch < 100000) {
      var i = 1
      var epochTrainTime = 0L
      while (i <= trainSize / batchSize) {
//        val index = Random.nextInt(735) + 1
        val startTime = System.nanoTime()
//        optm.optimize(feval(masterGrad, model, criterion, trainData.narrow(1, 1 + (index - 1) * batchSize, batchSize), trainLabel.narrow(1, 1 + (index - 1) * batchSize, batchSize)), masterWeights, config)
        optm.optimize(feval(masterGrad, model, criterion, trainData.narrow(1, 1 + (i - 1) * batchSize, batchSize), trainLabel.narrow(1, 1 + (i - 1) * batchSize, batchSize)), masterWeights, config, config)
        val endTime = System.nanoTime()
        epochTrainTime += (endTime - startTime)
        wallClockTime += (endTime - startTime)
        i += 1
//        evaluate(masterGrad, model, testData)
      }
      println(s"[$epoch]At training $correct of $count, ${correct.toDouble / count.toDouble}. Time cost ${(epochTrainTime) / 1e6 / trainSize.toDouble}ms for a single sample. Throughput is ${count.toDouble * 1e9 / (epochTrainTime)} records/second")
      println(s"Average Loss: ${l * batchSize / count}")

//      val o = model.forward(testData)
//      o.apply1(Math.exp(_))
//      println(o)

      l = 0.0
      correct = 0
      count = 0
      epoch += 1
    }
    evaluate(masterGrad, model, testData)
  }

  def feval(grad : Tensor[Double], module : Module[Double], criterion : Criterion[Double], input : Tensor[Double], target : Tensor[Double])(weights : Tensor[Double])
  : (Double, Tensor[Double]) = {
    module.training()
    grad.zero()
//    input.resize(Array(2))
    val output = module.forward(input)
//    val preRe = output.clone()
//    preRe.apply1(Math.exp(_))
//    println(s"${input(Array(1, 1))} ${input(Array(1, 2))} ${preRe}")
    for(d <- 1 to output.size(1)) {
//      val pre = maxIndex(output.select(1, d))
//      val pre = maxIndex(output)
//                print(s"|pre:${pre}tar:${target(Array(d))}|")
      val pre = Math.round(output(Array(d, 1)))
      count += 1
      if(pre == target.valueAt(d).toInt) correct += 1
    }

    val loss = criterion.forward(output, target)
    val gradOut = criterion.backward(output, target)
    module.backward(input, gradOut)
    //    print(module.gradInput)
    l += loss
    //    print(loss + " ")
    (loss, grad)
  }


  def evaluate(masterGrad : Tensor[Double], module : Module[Double], testData : Tensor[Double]) : Unit = {
//    module.evaluate()
    for(i <- 1 to testData.size(1)){
      masterGrad.zero()
      val input = testData.narrow(1, i, 1)
      input.resize(Array(2))
      val (pre, probs) = predictOur(input, module)
//      println(s"${input(Array(1))}\t${input(Array(2))}\t${pre}\t${probs(Array(1))}\t${probs(Array(2))}\t${probs(Array(3))} ")
      println(s"${input(Array(1))}\t${input(Array(2))}\t${pre}\t")
    }
  }

  def predictOur(input: Tensor[Double], model: Module[Double]) : (Int, Tensor[Double]) = {
    val logProbs = model.forward(input)
//    val probs = logProbs.apply1(logProb => Math.exp(logProb))
//    probs.resize(Array(3))
    println(logProbs(Array(1)))
//    return (maxIndex(probs), probs)
    return (Math.round(logProbs(Array(1))).toInt, logProbs)
  }

  def maxIndex(data: Tensor[Double]): Int = {
    require(data.dim() == 1)
    var index = 1
    val indexer = Array(1)
    var maxVal = data(indexer)
    var j = 2
    while (j <= data.size(1)) {
      indexer(0) = j
      if (data(indexer) > maxVal) {
        maxVal = data(indexer)
        index = j
      }
      j += 1
    }
    index
  }
}
