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

package com.intel.analytics.bigdl.dllib.nn

import com.intel.analytics.bigdl.ckks.CKKS
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.feature.FeatureSet
import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, Sample, SampleToMiniBatch, TensorSample}
import com.intel.analytics.bigdl.dllib.nn.ckks.{Encryptor, FusedBCECriterion}
import com.intel.analytics.bigdl.dllib.optim.{Adagrad, Ftrl, SGD}
import org.scalatest.FlatSpec
import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.dllib.utils.{Engine, RandomGenerator, T, TestUtils}
import com.intel.analytics.bigdl.dllib.utils.serializer.ModuleSerializationTest
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import java.util
import scala.collection.mutable.ArrayBuffer
import scala.math.abs
import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class SigmoidSpec extends FlatSpec {
  "A Sigmoid Module " should "generate correct output and grad" in {
    val module = new Sigmoid[Double]
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = 0.063364277360961
    input(Array(1, 1, 2)) = 0.90631252736785
    input(Array(1, 2, 1)) = 0.22275671223179
    input(Array(1, 2, 2)) = 0.37516756891273
    input(Array(2, 1, 1)) = 0.99284988618456
    input(Array(2, 1, 2)) = 0.97488326719031
    input(Array(2, 2, 1)) = 0.94414822547697
    input(Array(2, 2, 2)) = 0.68123375508003
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.38652365817688
    gradOutput(Array(1, 1, 2)) = 0.034144022269174
    gradOutput(Array(1, 2, 1)) = 0.68105488433503
    gradOutput(Array(1, 2, 2)) = 0.41517980070785
    gradOutput(Array(2, 1, 1)) = 0.91740695876069
    gradOutput(Array(2, 1, 2)) = 0.35317355184816
    gradOutput(Array(2, 2, 1)) = 0.24361599306576
    gradOutput(Array(2, 2, 2)) = 0.65869987895712
    val expectedOutput = Tensor[Double](2, 2, 2)
    expectedOutput(Array(1, 1, 1)) = 0.51583577126786
    expectedOutput(Array(1, 1, 2)) = 0.71224499952187
    expectedOutput(Array(1, 2, 1)) = 0.55546003768115
    expectedOutput(Array(1, 2, 2)) = 0.59270705262321
    expectedOutput(Array(2, 1, 1)) = 0.72965046058394
    expectedOutput(Array(2, 1, 2)) = 0.72609176575892
    expectedOutput(Array(2, 2, 1)) = 0.71993681755829
    expectedOutput(Array(2, 2, 2)) = 0.66401400310487
    val expectedGrad = Tensor[Double](2, 2, 2)
    expectedGrad(Array(1, 1, 1)) = 0.096533985368059
    expectedGrad(Array(1, 1, 2)) = 0.0069978877068295
    expectedGrad(Array(1, 2, 1)) = 0.16816892172375
    expectedGrad(Array(1, 2, 2)) = 0.1002266468557
    expectedGrad(Array(2, 1, 1)) = 0.18096830763559
    expectedGrad(Array(2, 1, 2)) = 0.070240043677749
    expectedGrad(Array(2, 2, 1)) = 0.049119755820981
    expectedGrad(Array(2, 2, 2)) = 0.14695555224503
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    expectedOutput.map(output, (v1, v2) => {
      TestUtils.conditionFailTest(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedGrad.map(gradInput, (v1, v2) => {
      TestUtils.conditionFailTest(abs(v1 - v2) < 1e-6);
      v1
    })
    TestUtils.conditionFailTest(input == inputOrg)
    TestUtils.conditionFailTest(gradOutput == gradOutputOrg)
  }

  "ckks layer" should "generate correct output and grad" in {
    val eps = 1e-12f
    val module = new Sigmoid[Float]
    val criterion = new BCECriterion[Float]()

    val input = Tensor[Float](2, 2)
    input(Array(1, 1)) = 0.063364277360961f
    input(Array(1, 2)) = 0.90631252736785f
    input(Array(2, 1)) = 0.22275671223179f
    input(Array(2, 2)) = 0.37516756891273f
    val target = Tensor[Float](2, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 0
    target(Array(2, 2)) = 1

    val exceptedOutput = module.forward(input)

    val exceptedLoss = criterion.forward(exceptedOutput, target)
    val exceptedGradOutput = criterion.backward(exceptedOutput, target)
    val exceptedGradInput = module.backward(input, exceptedGradOutput)


    val ckks = new CKKS()
    val secrets = ckks.createSecrets()
    val encryptorPtr = ckks.createCkksEncryptor(secrets)
    val ckksRunnerPtr = ckks.createCkksCommonInstance(secrets)
    val enTarget =
      Tensor[Byte](Storage[Byte](ckks.ckksEncrypt(encryptorPtr, target.storage().array())))
        .resize(target.size())

    val module2 = Encryptor[Float](encryptorPtr)
    val criterion2 = FusedBCECriterion(ckksRunnerPtr)

    val output2 = module2.forward(input).toTensor[Byte]
    val loss2 = criterion2.forward(output2, enTarget)
    val gradOutput2 = criterion2.backward(output2, enTarget)
    val gradInput2 = module2.backward(input, gradOutput2).toTensor[Float]


    val enLoss = ckks.ckksDecrypt(encryptorPtr, loss2.storage().array)
    gradInput2.div(4)
    val loss = enLoss.slice(0, 4).sum / 4
    println(loss + "  "  + exceptedLoss)
    println(gradInput2)
    println(exceptedGradInput)
  }

  "ckks" should "generate correct output and grad" in {
    val eps = 1e-12f
    val module = new Sigmoid[Float]
    val criterion = new BCECriterion[Float]()
    val input = Tensor[Float](2, 2)
    input(Array(1, 1)) = 0.063364277360961f
    input(Array(1, 2)) = 0.90631252736785f
    input(Array(2, 1)) = 0.22275671223179f
    input(Array(2, 2)) = 0.37516756891273f
    val target = Tensor[Float](2, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 0
    target(Array(2, 2)) = 1

//    val manuO = new Array[Float](8)
//    (0 until 8).foreach{i =>
//      val in = input.storage().array()(i)
//      val tar = target.storage().array()(i)
//      manuO(i) = (- tar * math.log(in) + (tar - 1) * math.log(1 - in)).toFloat
//    }

//    val oo = target.clone().add(input.clone().log()) +
//    target.clone().mul(-1).add(1).cmul(input.clone().mul(-1).add(1).log())

    val exceptedOutput = module.forward(input)

    val ooo = target.clone().cmul(exceptedOutput.clone().add(eps).log()).add(
      target.clone().mul(-1).add(1).cmul(exceptedOutput.clone().mul(-1).add(1).add(eps).log()))
    ooo.mul(-1)
    val eo = ooo.sum() / 4


    val exceptedLoss = criterion.forward(exceptedOutput, target)
    val exceptedGradInput1 = criterion.backward(exceptedOutput, target)
    val exceptedGradInput2 = module.backward(input, exceptedGradInput1)


    val ckks = new CKKS()
    val secrets = ckks.createSecrets()
    val encryptorPtr = ckks.createCkksEncryptor(secrets)
    val ckksRunnerPtr = ckks.createCkksCommonInstance(secrets)
    val enInput = ckks.ckksEncrypt(encryptorPtr, input.storage().array())
    val enTarget = ckks.ckksEncrypt(encryptorPtr, target.storage().array())
    val o = ckks.train(ckksRunnerPtr, enInput, enTarget)
    val enLoss = ckks.ckksDecrypt(encryptorPtr, o(0))
    val enGradInput2 = ckks.ckksDecrypt(encryptorPtr, o(1))
    val gradInput2 = Tensor[Float](enGradInput2.slice(0, 4), Array(2, 2))
    gradInput2.div(4)
    val loss = enLoss.slice(0, 4).sum / 4
    println(loss + "  "  + exceptedLoss)
    println(gradInput2)
    println()
  }

  "ckks forward" should "generate correct output" in {
    val module = new Sigmoid[Float]
    val input = Tensor[Float](2, 2, 2)
    input(Array(1, 1, 1)) = 0.063364277360961f
    input(Array(1, 1, 2)) = 0.90631252736785f
    input(Array(1, 2, 1)) = 0.22275671223179f
    input(Array(1, 2, 2)) = 0.37516756891273f
    input(Array(2, 1, 1)) = 0.99284988618456f
    input(Array(2, 1, 2)) = 0.97488326719031f
    input(Array(2, 2, 1)) = 0.94414822547697f
    input(Array(2, 2, 2)) = 0.68123375508003f
    val exceptedOutput = module.forward(input)

    val ckks = new CKKS()
    val secrets = ckks.createSecrets()
    val encryptorPtr = ckks.createCkksEncryptor(secrets)
    val ckksRunnerPtr = ckks.createCkksCommonInstance(secrets)
    val enInput = ckks.ckksEncrypt(encryptorPtr, input.storage().array())
    val enOutput = ckks.sigmoidForward(ckksRunnerPtr, enInput)
    val outputArray = ckks.ckksDecrypt(encryptorPtr, enOutput(0))
    val output = Tensor[Float](outputArray.slice(0, 8), Array(2, 2, 2))
    println(output)
    println(exceptedOutput)
  }

  "ckks train" should "converge" in {
    Engine.init(1, 1, false)
    val random = new Random()
    random.setSeed(10)
    val featureLen = 10
    val bs = 20
    val totalSize = 1000
    val dummyData = Array.tabulate(totalSize)(i =>
      {
        val features = Array.tabulate(featureLen)(_ => random.nextFloat())
        val label = math.round(features.sum / featureLen).toFloat
        Sample[Float](Tensor[Float](features, Array(featureLen)), label)
      }
    )
    val dataset = DataSet.array(dummyData) -> SampleToMiniBatch[Float](bs)

    val module = Sequential[Float]()
    module.add(Linear[Float](10, 1))
    module.add(Sigmoid[Float]())
    val criterion = new BCECriterion[Float]()
    val sgd = new SGD[Float](0.1)
//    val sgd2 = new SGD[Float](0.5, learningRateDecay = 1e-4)
    val sgd2 = new SGD[Float](0.1)
    val (weight, gradient) = module.getParameters()

    val module2 = Linear[Float](10, 1)
    val sigmoid2 = Sigmoid[Float]()
    val (weight2, gradient2) = module2.getParameters()
    weight2.copy(weight)
    val ckks = new CKKS()
    val secrets = ckks.createSecrets()
    val encryptorPtr = ckks.createCkksEncryptor(secrets)
    val ckksRunnerPtr = ckks.createCkksCommonInstance(secrets)

    var a = 0
    val epochNum = 20
    val lossArray = new Array[Float](epochNum)
    val loss2Array = new Array[Float](epochNum)
    (0 until epochNum).foreach{epoch =>
      var countLoss = 0f
      var countLoss2 = 0f
      dataset.shuffle()
      val trainData = dataset.toLocal().data(false)
      while(trainData.hasNext) {
        val miniBatch = trainData.next()
        val input = miniBatch.getInput()
        val target = miniBatch.getTarget()
        val output = module.forward(input)
        val loss = criterion.forward(output, target)
        countLoss += loss
        if (a < 4) {
          a += 1
          println(countLoss / a)
        }
        val gradOutput = criterion.backward(output, target)
        module.backward(input, gradOutput)
        sgd.optimize(_ => (loss, gradient), weight)

//        val output2 = sigmoid2.forward(module2.forward(input).toTensor[Float])
//        val enInput = ckks.ckksEncrypt(encryptorPtr, output2.storage().array())
//        val enTarget = ckks.ckksEncrypt(encryptorPtr, target.toTensor[Float].storage().array())
//        val o = ckks.backward(ckksRunnerPtr, enInput, enTarget)

        val output2 = module2.forward(input).toTensor[Float]
        val enInput = ckks.ckksEncrypt(encryptorPtr, output2.storage().array())
        val enTarget = ckks.ckksEncrypt(encryptorPtr, target.toTensor[Float].storage().array())
        val o = ckks.train(ckksRunnerPtr, enInput, enTarget)

        val enLoss = ckks.ckksDecrypt(encryptorPtr, o(0))
        val enGradInput2 = ckks.ckksDecrypt(encryptorPtr, o(1))
        val gradInput2 = Tensor[Float](enGradInput2.slice(0, bs), Array(bs, 1))
        gradInput2.div(bs)
        module2.backward(input, gradInput2)
        val loss2 = enLoss.slice(0, bs).sum / bs
        sgd2.optimize(_ => (loss2, gradient2), weight2)
        countLoss2 += loss2
        if (a < 4) {
          println(countLoss2 / a)
        }
        module.zeroGradParameters()
        module2.zeroGradParameters()
      }
      lossArray(epoch) = countLoss / (totalSize / bs)
      loss2Array(epoch) = countLoss2 / (totalSize / bs)
      println(countLoss / (totalSize / bs))
      println("           " + countLoss2 / (totalSize / bs))
    }
    println("loss1: ")
    println(lossArray.mkString("\n"))
    println("loss2: ")
    println(loss2Array.mkString("\n"))


  }


  "train breast" should "work" in {
    import org.apache.spark.sql
    import org.apache.spark.sql._
    import org.apache.spark.sql.types._
    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.mllib.linalg.{Vector, Vectors}
    import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
    import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
    /**
     *  Missing values show up as a dot.  The dot function below
     *  returns -1 if there is a dot or blank value. And it converts strings to double.
     * Then later we will
     *  delete all rows that have any -1 values.
     */
    def dot (s: String) : Double = {
      if (s.contains(".") || s.length == 0) {
        return -1
      } else {
        return s.toDouble
      }
    }
    /**
     *  We are going to use a Dataframe.  It requires a schema.
     * So we create that below.  We use the same column names
     *  as are in the .dat file.
     */
    val schema = StructType (
      StructField("STR", DoubleType, true) ::
        StructField("OBS", DoubleType, true) ::
        StructField("AGMT", DoubleType, true) ::
        StructField("FNDX", DoubleType, true) ::
        StructField("HIGD", DoubleType, true) ::
        StructField("DEG", DoubleType, true) ::
        StructField("CHK", DoubleType, true) ::
        StructField("AGP1", DoubleType, true) ::
        StructField("AGMN", DoubleType, true) ::
        StructField("NLV", DoubleType, true) ::
        StructField("LIV", DoubleType, true) ::
        StructField("WT", DoubleType, true) ::
        StructField("AGLP", DoubleType, true) ::
        StructField("MST", DoubleType, true) :: Nil)
    /**
     *  Read in the .dat file and use the regular expression
     *  \s+ to split it by spaces into an RDD.
     */
      val conf = new SparkConf().setMaster("local[1]").setAppName("breast")
      val sc = new SparkContext(conf)
    Engine.init(1, 1, true)
    val spark = SparkSession.builder().getOrCreate()
    val readingsRDD = sc.textFile("/home/xin/datasets/breast/data.dat")
    val RDD = readingsRDD.map(_.split("\\s+"))
    /**
     *   Run the dot function over every element in the RDD to convert them
     *   to doubles, since that if the format requires by the Spark ML LR model.
     *   Note that we skip the first one since that is just a blank space.
     */
    val rowRDD = RDD.map(s => Row(dot(s(1)), dot(s(2)), dot(s(3)), dot(s(4)), dot(s(5)), dot(s(6)),
      dot(s(7)), dot(s(8)), dot(s(9)), dot(s(10)), dot(s(11)), dot(s(12)),
      dot(s(13)), dot(s(14))))
    /**
     * Now create a dataframe with the schema we described above,
     *
     */
    val readingsDF = spark.createDataFrame(rowRDD, schema)
    /**
     *  Create a new dataframe dropping all of those with missing values.
     */
    var cleanDF = readingsDF.filter(readingsDF("STR") > -1 && readingsDF("OBS") > -1 && readingsDF("AGMT")  > -1  && readingsDF("FNDX") > -1 && readingsDF("HIGD") > -1  && readingsDF("DEG") > -1 && readingsDF("CHK") > -1 && readingsDF("AGP1") > -1  && readingsDF("AGMN") > -1  && readingsDF("NLV") > -1  && readingsDF("LIV") > -1 && readingsDF("WT") > -1 && readingsDF("AGLP") > -1 && readingsDF("MST") > -1)
    /**
     *  Now comes something more complicated.  Our dataframe has the column headings
     *  we created with the schema.  But we need a column called “label” and one called
     * “features” to plug into the LR algorithm.  So we use the VectorAssembler() to do that.
     * Features is a Vector of doubles.  These are all the values like patient age, etc. that
     * we extracted above.  The label indicated whether the patient has cancer.
     */
    val featureCols = Array("STR", "OBS", "AGMT", "HIGD", "DEG", "CHK", "AGP1", "AGMN", "NLV", "LIV", "WT", "AGLP", "MST")
//    val featureCols = Array("STR", "OBS", "HIGD", "DEG", "CHK", "AGP1", "AGMN", "NLV", "LIV", "MST")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(cleanDF)
    /**
     * Then we use the StringIndexer to take the column FNDX and make that the label.
     *  FNDX is the 1 or 0 indicator that shows whether the patient has cancer.
     * Like the VectorAssembler it will add another column to the dataframe.
     */
      def preprocess(features: Array[Float]): Array[Float] = {
        (0 until features.length).foreach{i =>
          features(i) = features(i) / 100
        }
        features
      }
    val labelIndexer = new StringIndexer().setInputCol("FNDX").setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2)
    val numFeature = featureCols.length
    val trainData = df3.collect().map{d =>
      val feature = Tensor[Float](preprocess(d.getAs[DenseVector](14).values.map(_.toFloat)),
        Array(numFeature))
      val label = d.getDouble(15).toFloat
      Sample[Float](feature, label)
    }
    val bs = 10
    val totalSize = trainData.size
    val dataset = DataSet.array(trainData) -> SampleToMiniBatch[Float](bs)

    val module = Sequential[Float]()
    module.add(Linear[Float](numFeature, 1))
    module.add(Sigmoid[Float]())
    val criterion = new BCECriterion[Float]()
    val sgd = new Adagrad[Float](0.02)
    //    val sgd2 = new SGD[Float](0.5, learningRateDecay = 1e-4)
    val sgd2 = new Adagrad[Float](0.02)
    val (weight, gradient) = module.getParameters()

    val module2 = Linear[Float](numFeature, 1)
    val sigmoid2 = Sigmoid[Float]()
    val (weight2, gradient2) = module2.getParameters()
    weight2.copy(weight)
    val ckks = new CKKS()
    val secrets = ckks.createSecrets()
    val encryptorPtr = ckks.createCkksEncryptor(secrets)
    val ckksRunnerPtr = ckks.createCkksCommonInstance(secrets)

    var a = 0
    val epochNum = 40
    val lossArray = new Array[Float](epochNum)
    val loss2Array = new Array[Float](epochNum)
    (0 until epochNum).foreach{epoch =>
      var countLoss = 0f
      var countLoss2 = 0f
      dataset.shuffle()
      val trainData = dataset.toLocal().data(false)
      while(trainData.hasNext) {
        val miniBatch = trainData.next()
        val input = miniBatch.getInput()
        val currentBs = input.toTensor[Float].size(1)
        val target = miniBatch.getTarget()
        val output = module.forward(input)
        val loss = criterion.forward(output, target)
        countLoss += loss
        if (a < 4) {
          a += 1
          println(countLoss / a)
        }
        val gradOutput = criterion.backward(output, target)
        module.backward(input, gradOutput)
        sgd.optimize(_ => (loss, gradient), weight)

//        val output2 = module2.forward(input).toTensor[Float]
//        val enInput = ckks.ckksEncrypt(encryptorPtr, output2.storage().array())
//        val enTarget = ckks.ckksEncrypt(encryptorPtr, target.toTensor[Float].storage().array())
//        val o = ckks.train(ckksRunnerPtr, enInput, enTarget)

        val output2 = sigmoid2.forward(module2.forward(input).toTensor[Float])
        val enInput = ckks.ckksEncrypt(encryptorPtr, output2.storage().array())
        val enTarget = ckks.ckksEncrypt(encryptorPtr, target.toTensor[Float].storage().array())
        val o = ckks.backward(ckksRunnerPtr, enInput, enTarget)

        val enLoss = ckks.ckksDecrypt(encryptorPtr, o(0))
        val enGradInput2 = ckks.ckksDecrypt(encryptorPtr, o(1))
        val gradInput2 = Tensor[Float](enGradInput2.slice(0, currentBs), Array(currentBs, 1))
        gradInput2.div(currentBs)
        module2.backward(input, gradInput2)
        val loss2 = enLoss.slice(0, currentBs).sum / currentBs
        sgd2.optimize(_ => (loss2, gradient2), weight2)
        countLoss2 += loss2
        if (a < 4) {
          println(countLoss2 / a)
        }
        module.zeroGradParameters()
        module2.zeroGradParameters()
      }
      lossArray(epoch) = countLoss / Math.ceil(totalSize / bs).toFloat
      loss2Array(epoch) = countLoss2 / Math.ceil(totalSize / bs).toFloat
      println(countLoss / (totalSize / bs))
      println("           " + countLoss2 / (totalSize / bs))
    }
    println("loss1: ")
    println(lossArray.mkString("\n"))
    println("loss2: ")
    println(loss2Array.mkString("\n"))

    (0 until epochNum).foreach{epoch =>
      module.evaluate()
      module2.evaluate()
      val evalData = dataset.toLocal().data(false)
      while( evalData.hasNext) {
        val miniBatch = evalData.next()
        val input = miniBatch.getInput()
        val currentBs = input.toTensor[Float].size(1)
        val target = miniBatch.getTarget().toTensor[Float]
        val pre = module.forward(input)

        val output2 = module2.forward(input).toTensor[Float]
        val enInput = ckks.ckksEncrypt(encryptorPtr, output2.storage().array())
        val enPre = ckks.sigmoidForward(ckksRunnerPtr, enInput)
        val pre2 = ckks.ckksDecrypt(encryptorPtr, enPre(0))

        println("target dllib ckks")
        (0 until currentBs).foreach{i =>
          val dllibPre = pre.toTensor[Float].valueAt(i + 1, 1)
          val ckksPre = pre2(i)
          println(target.valueAt(i + 1, 1) + " " + dllibPre + " " + ckksPre)

        }
      }
    }
  }

  "census" should "converge" in {
    import org.apache.logging.log4j.{Level, LogManager}
    import org.apache.logging.log4j.core.config.Configurator
    import org.apache.spark.SparkConf
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.{DataFrame, SQLContext}
    import org.apache.spark.rdd.RDD
    import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

    import scala.reflect.ClassTag

    case class Record(
                       age: Int,
                       workclass: String,
                       fnlwgt: Int,
                       education: String,
                       education_num: Int,
                       marital_status: String,
                       occupation: String,
                       relationship: String,
                       race: String,
                       gender: String,
                       capital_gain: Int,
                       capital_loss: Int,
                       hours_per_week: Int,
                       native_country: String,
                       income_bracket: String
                     )


    val recordSchema = StructType(Array(
      StructField("age", IntegerType, false),
      StructField("workclass", StringType, false),
      StructField("fnlwgt", IntegerType, false),
      StructField("education", StringType, false),
      StructField("education_num", IntegerType, false),
      StructField("marital_status", StringType, false),
      StructField("occupation", StringType, false),
      StructField("relationship", StringType, false),
      StructField("race", StringType, false),
      StructField("gender", StringType, false),
      StructField("capital_gain", IntegerType, false),
      StructField("capital_loss", IntegerType, false),
      StructField("hours_per_week", IntegerType, false),
      StructField("native_country", StringType, false),
      StructField("income_bracket", StringType, false)
    ))

    case class RecordSample[T: ClassTag](sample: Sample[T])
    Configurator.setLevel("org", Level.ERROR)
    def loadCensusData(spark: SparkSession, dataPath: String): (DataFrame, DataFrame) = {
      val training = spark.sparkContext
        .textFile(dataPath + "/adult.data")
        .map(_.split(",").map(_.trim))
        .filter(_.size == 15).map(array =>
        Row(
          array(0).toInt, array(1), array(2).toInt, array(3), array(4).toInt,
          array(5), array(6), array(7), array(8), array(9),
          array(10).toInt, array(11).toInt, array(12).toInt, array(13), array(14)
        )
      )

      val validation = spark.sparkContext
        .textFile(dataPath + "/adult.test")
        .map(_.dropRight(1))  // remove dot at the end of each line in adult.test
        .map(_.split(",").map(_.trim))
        .filter(_.size == 15).map(array =>
        Row(
          array(0).toInt, array(1), array(2).toInt, array(3), array(4).toInt,
          array(5), array(6), array(7), array(8), array(9),
          array(10).toInt, array(11).toInt, array(12).toInt, array(13), array(14)
        )
      )

      (spark.createDataFrame(training, recordSchema),
        spark.createDataFrame(validation, recordSchema))
    }
    case class ColumnFeatureInfo(wideBaseCols: Array[String] = Array[String](),
                                 wideBaseDims: Array[Int] = Array[Int](),
                                 wideCrossCols: Array[String] = Array[String](),
                                 wideCrossDims: Array[Int] = Array[Int](),
                                 indicatorCols: Array[String] = Array[String](),
                                 indicatorDims: Array[Int] = Array[Int](),
                                 embedCols: Array[String] = Array[String](),
                                 embedInDims: Array[Int] = Array[Int](),
                                 embedOutDims: Array[Int] = Array[Int](),
                                 continuousCols: Array[String] = Array[String](),
                                 label: String = "label") extends Serializable{
      override def toString: String = {
        "wideBaseCols:" + wideBaseCols.mkString(",") +"\n" +
          "wideBaseDims:" + wideBaseDims.mkString(",") + "\n" +
          "wideCrossCols:" + wideCrossCols.mkString(",") +"\n" +
          "wideCrossDims:" + wideCrossDims.mkString(",") +"\n" +
          "indicatorCols:" + indicatorCols.mkString(",") + "\n" +
          "indicatorDims:" + indicatorDims.mkString(",") +"\n" +
          "embedCols:" + embedCols.mkString(",") +"\n" +
          "embedInDims:" + embedInDims.mkString(",") +"\n" +
          "embedOutDims:" + embedOutDims.mkString(",") +"\n" +
          "continuousCols:" + continuousCols.mkString(",") +"\n" +
          "label:" + label

      }
    }
    def categoricalFromVocabList(vocabList: Array[String]): (String) => Int = {
      val func = (sth: String) => {
        val default: Int = 0
        val start: Int = 1
        if (vocabList.contains(sth)) vocabList.indexOf(sth) + start
        else default
      }
      func
    }
    def buckBuckets(bucketSize: Int)(col: String*): Int = {
      Math.abs(col.reduce(_ + "_" + _).hashCode()) % bucketSize + 0
    }

    def bucketizedColumn(boundaries: Array[Float]): Float => Int = {
      col1: Float => {
        var index = 0
        while (index < boundaries.length && col1 >= boundaries(index)) {
          index += 1
        }
        index
      }
    }

    def getDeepTensor(r: Row, columnInfo: ColumnFeatureInfo): Tensor[Float] = {
      val deepColumns1 = columnInfo.indicatorCols
      val deepColumns2 = columnInfo.embedCols ++ columnInfo.continuousCols
      val deepLength = columnInfo.indicatorDims.sum + deepColumns2.length
      val deepTensor = Tensor[Float](deepLength).fill(0)

      // setup indicators
      var acc = 0
      (0 to deepColumns1.length - 1).map {
        i =>
          val index = r.getAs[Int](columnInfo.indicatorCols(i))
          val accIndex = if (i == 0) index
          else {
            acc = acc + columnInfo.indicatorDims(i - 1)
            acc + index
          }
          deepTensor.setValue(accIndex + 1, 1)
      }

      // setup embedding and continuous
      (0 to deepColumns2.length - 1).map {
        i =>
          deepTensor.setValue(i + 1 + columnInfo.indicatorDims.sum,
            r.getAs[Int](deepColumns2(i)).toFloat)
      }
      deepTensor
    }

    def getWideTensor(r: Row, columnInfo: ColumnFeatureInfo): Tensor[Float] = {
      val wideColumns = columnInfo.wideBaseCols ++ columnInfo.wideCrossCols
      val wideDims = columnInfo.wideBaseDims ++ columnInfo.wideCrossDims
      val wideLength = wideColumns.length
      var acc = 0
      val indices: Array[Int] = (0 to wideLength - 1).map(i => {
        val index = r.getAs[Int](wideColumns(i))
        if (i == 0) {index}
        else {
          acc = acc + wideDims(i - 1)
          acc + index
        }
      }).toArray
      val values = indices.map(_ => 1.0f)
      val shape = Array(wideDims.sum)

      Tensor.sparse(Array(indices), values, shape)
    }

    def getWideTensorSequential(r: Row, columnInfo: ColumnFeatureInfo): Tensor[Float] = {
      val wideColumns = columnInfo.wideBaseCols ++ columnInfo.wideCrossCols
      val wideDims = columnInfo.wideBaseDims ++ columnInfo.wideCrossDims
      val wideLength = wideColumns.length
      var acc = 0
      val indices: Array[Int] = (0 to wideLength - 1).map(i => {
        val index = r.getAs[Int](wideColumns(i))
        if (i == 0) index
        else {
          acc = acc + wideDims(i - 1)
          acc + index
        }
      }).toArray
      val values = indices.map(_ => 1.0f)
      val shape = Array(wideDims.sum)

      Tensor.sparse(Array(indices), values, shape)
    }

    def row2SampleSequential(r: Row,
                             columnInfo: ColumnFeatureInfo,
                             modelType: String): Sample[Float] = {
      val wideTensor: Tensor[Float] = getWideTensorSequential(r, columnInfo)
      val deepTensor: Tensor[Float] = getDeepTensor(r, columnInfo)
      val l = r.getAs[Int](columnInfo.label)

      val label = Tensor[Float](T(l))
      label.resize(1)

      modelType match {
        case "wide_n_deep" =>
          TensorSample[Float](Array(wideTensor, deepTensor), Array(label))
        case "wide" =>
          TensorSample[Float](Array(wideTensor), Array(label))
        case "deep" =>
          TensorSample[Float](Array(deepTensor), Array(label))
        case _ =>
          throw new IllegalArgumentException("unknown type")
      }
    }

    // convert features to RDD[Sample[Float]]
    def assemblyFeature(isImplicit: Boolean = false,
                        dataDf: DataFrame,
                        columnInfo: ColumnFeatureInfo,
                        modelType: String): RDD[RecordSample[Float]] = {
      val educationVocab = Array("Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th") // 16
      val maritalStatusVocab = Array("Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed")
      val relationshipVocab = Array("Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative") // 6
      val workclassVocab = Array("Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked") // 9
      val genderVocab = Array("Female", "Male")

      val ages = Array(18f, 25, 30, 35, 40, 45, 50, 55, 60, 65)

      val educationVocabUdf = udf(categoricalFromVocabList(educationVocab))
      val maritalStatusVocabUdf = udf(categoricalFromVocabList(maritalStatusVocab))
      val relationshipVocabUdf = udf(categoricalFromVocabList(relationshipVocab))
      val workclassVocabUdf = udf(categoricalFromVocabList(workclassVocab))
      val genderVocabUdf = udf(categoricalFromVocabList(genderVocab))

      val bucket1Udf = udf(buckBuckets(1000)(_: String))
      val bucket2Udf = udf(buckBuckets(1000)(_: String, _: String))
      val bucket3Udf = udf(buckBuckets(1000)(_: String, _: String, _: String))

      val ageBucketUdf = udf(bucketizedColumn(ages))

      val incomeUdf = udf((income: String) => if (income == ">50K" || income == ">50K.") 1 else 0)

      val data = dataDf
        .withColumn("age_bucket", ageBucketUdf(col("age")))
        .withColumn("edu_occ", bucket2Udf(col("education"), col("occupation")))
        .withColumn("age_edu_occ", bucket3Udf(col("age_bucket"), col("education"),
          col("occupation")))
        .withColumn("edu", educationVocabUdf(col("education")))
        .withColumn("mari", maritalStatusVocabUdf(col("marital_status")))
        .withColumn("rela", relationshipVocabUdf(col("relationship")))
        .withColumn("work", workclassVocabUdf(col("workclass")))
        .withColumn("occ", bucket1Udf(col("occupation")))
        .withColumn("label", incomeUdf(col("income_bracket")))

      val rddOfSample = data.rdd.map(r => {
        RecordSample(row2SampleSequential(r, columnInfo, modelType))
      })
      rddOfSample
    }

    val batchSize = 8192
    val maxEpoch = 10
    val modelType = "wide"
    val inputDir = "/home/xin/datasets/adult"

    val conf = new SparkConf().setAppName("WideAndDeepExample").setMaster("local[4]")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SparkSession.builder().getOrCreate()

    val (trainDf, valDf) = loadCensusData(sqlContext, inputDir)

    println(trainDf.show(10))

    val localColumnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("edu", "mari", "rela", "work", "occ", "age_bucket"),
      wideBaseDims = Array(16, 7, 6, 9, 1000, 11),
      wideCrossCols = Array("edu_occ", "age_edu_occ"),
      wideCrossDims = Array(1000, 1000),
      indicatorCols = Array("work", "edu", "mari", "rela"),
      indicatorDims = Array(9, 16, 7, 6),
      embedCols = Array("occ"),
      embedInDims = Array(1000),
      embedOutDims = Array(8),
      continuousCols = Array("age", "education_num", "capital_gain",
        "capital_loss", "hours_per_week"))

    val isImplicit = false
    val trainpairFeatureRdds =
      assemblyFeature(isImplicit, trainDf, localColumnInfo, modelType)

    val sample1 = trainpairFeatureRdds.take(10)

    val validationpairFeatureRdds =
      assemblyFeature(isImplicit, valDf, localColumnInfo, modelType)

    val optimMethods = if (modelType == "wide_n_deep") {
      Map("deepPart" -> new Adagrad[Float](0.001),
        "widePart" -> new Ftrl[Float](math.min(5e-3, 1 / math.sqrt(3049))))
    } else if (modelType == "wide") {
      Map("widePart" -> new Ftrl[Float](math.min(5e-3, 1 / math.sqrt(3049))))
    } else if (modelType == "deep") {
      Map("deepPart" -> new Adagrad[Float](0.001))
    } else {
      throw new IllegalArgumentException(s"Unkown modelType ${modelType}")
    }

    val sample2batch = SampleToMiniBatch[Float](batchSize)
//    val trainRdds = FeatureSet.rdd(trainpairFeatureRdds.map(x => x.sample)) ->
//      sample2batch
//    val validationRdds = FeatureSet.rdd(validationpairFeatureRdds.map(x => x.sample)) ->
//      sample2batch
    val trainDataset = DataSet.array(
        trainpairFeatureRdds.map(_.sample).collect()) -> SampleToMiniBatch[Float](batchSize)
    val validationDataset = DataSet.array(
        validationpairFeatureRdds.map(_.sample).collect()) -> SampleToMiniBatch[Float](batchSize)

    val numFeature = 3049
    val totalSize = trainDataset.size()
    val bs = batchSize

    RandomGenerator.RNG.setSeed(2L)
    val linear = SparseLinear[Float](numFeature, 1)
    val sigmoid = Sigmoid[Float]()
    val criterion = new BCECriterion[Float]()
    val sgd = new Adagrad[Float](0.005)
    //    val sgd2 = new SGD[Float](0.5, learningRateDecay = 1e-4)
    val sgd2 = new Adagrad[Float](0.005)
    val (weight, gradient) = linear.getParameters()
    weight.randn(0, 0.001)

    val linear2 = SparseLinear[Float](numFeature, 1)
    val (weight2, gradient2) = linear2.getParameters()
    weight2.copy(weight)
    val ckks = new CKKS()
    val secrets = ckks.createSecrets()
    val encryptorPtr = ckks.createCkksEncryptor(secrets)
    val ckksRunnerPtr = ckks.createCkksCommonInstance(secrets)

    var a = 0
    val epochNum = 40
    val lossArray = new Array[Float](epochNum)
    val loss2Array = new Array[Float](epochNum)
    var ckksTime = Array(0L, 0L, 0L)
    var dllibTime = Array(0L, 0L, 0L)
    (0 until epochNum).foreach{epoch =>
      var countLoss = 0f
      var countLoss2 = 0f
      trainDataset.shuffle()
      val trainData = trainDataset.toLocal().data(false)
      while(trainData.hasNext) {
        val miniBatch = trainData.next()
        val input = miniBatch.getInput()
        val currentBs = input.toTensor[Float].size(1)
        val target = miniBatch.getTarget()

        val dllibStart = System.nanoTime()
        val doutput1 = linear.forward(input)
        dllibTime(1) += System.nanoTime() - dllibStart
        val doutput2 = sigmoid.forward(doutput1)
        val loss = criterion.forward(doutput2, target)
        val gradOutput = criterion.backward(doutput2, target)
        val dgradInput1 = sigmoid.backward(doutput1, gradOutput)
        val dllibLstart = System.nanoTime()
        linear.backward(input, dgradInput1)
        dllibTime(1) += System.nanoTime() - dllibLstart
        val updateStart = System.nanoTime()
        sgd.optimize(_ => (loss, gradient), weight)
        linear.zeroGradParameters()
        dllibTime(2) += System.nanoTime() - updateStart
        dllibTime(0) += System.nanoTime() - dllibStart
//        ckksTime(0) += criterion

        countLoss += loss
        if (a < 4) {
          a += 1
          println(countLoss / a)
        }

        val ckksStart = System.nanoTime()
        val output2 = linear2.forward(input).toTensor[Float]
        ckksTime(1) += System.nanoTime() - ckksStart
        val enInput = ckks.ckksEncrypt(encryptorPtr, output2.storage().array())
        val enTarget = ckks.ckksEncrypt(encryptorPtr, target.toTensor[Float].storage().array())
        val o = ckks.train(ckksRunnerPtr, enInput, enTarget)

//        val output2 = sigmoid2.forward(module2.forward(input).toTensor[Float])
//        val enInput = ckks.ckksEncrypt(encryptorPtr, output2.storage().array())
//        val enTarget = ckks.ckksEncrypt(encryptorPtr, target.toTensor[Float].storage().array())
//        val o = ckks.backward(ckksRunnerPtr, enInput, enTarget)

        val enLoss = ckks.ckksDecrypt(encryptorPtr, o(0))
        val enGradInput2 = ckks.ckksDecrypt(encryptorPtr, o(1))
        val gradInput2 = Tensor[Float](enGradInput2.slice(0, currentBs), Array(currentBs, 1))
        gradInput2.div(currentBs)
        val ckksLstart = System.nanoTime()
        linear2.backward(input, gradInput2)
        val loss2 = enLoss.slice(0, currentBs).sum / currentBs
        ckksTime(1) += System.nanoTime() - ckksLstart

        val updateStart2 = System.nanoTime()
        sgd2.optimize(_ => (loss2, gradient2), weight2)
        linear2.zeroGradParameters()
        ckksTime(2) += System.nanoTime() - updateStart2
        ckksTime(0) += System.nanoTime() - ckksStart
        countLoss2 += loss2
        if (a < 4) {
          println(countLoss2 / a)
        }
      }
      lossArray(epoch) = countLoss / Math.ceil(totalSize / bs).toFloat
      loss2Array(epoch) = countLoss2 / Math.ceil(totalSize / bs).toFloat
      println(countLoss / (totalSize / bs))
      println("           " + countLoss2 / (totalSize / bs))
      println(s"ckksTime: " + ckksTime(0) / 1e9)
      println(s"dllibTime: " + dllibTime(0) / 1e9)
      println(s"ckks Linear Time: ${ckksTime(1) / 1e9}")
      println(s"dllib Linear Time: ${dllibTime(1) / 1e9}")
      println(s"ckks update Time: ${ckksTime(2) / 1e9}")
      println(s"dllib update Time: ${dllibTime(2) / 1e9}")
      util.Arrays.fill(ckksTime, 0L)
      util.Arrays.fill(dllibTime, 0L)
    }
    println("loss1: ")
    println(lossArray.mkString("\n"))
    println("loss2: ")
    println(loss2Array.mkString("\n"))


    println(trainpairFeatureRdds.count())
    println(validationpairFeatureRdds.count())

    linear.evaluate()
    linear2.evaluate()
    val evalData = validationDataset.toLocal().data(false)
    var accDllib = 0
    var accCkks = 0
    var ckksPreTime = Array(0L, 0L)
    var dllibPreTime = Array(0L, 0L)
    while( evalData.hasNext) {
      val miniBatch = evalData.next()
      val input = miniBatch.getInput()
      val currentBs = input.toTensor[Float].size(1)
      val target = miniBatch.getTarget().toTensor[Float]
      val dllibStart = System.nanoTime()
      val output = linear.forward(input)
      dllibPreTime(1) += System.nanoTime() - dllibStart
      val pre = sigmoid.forward(output)
      dllibPreTime(0) += System.nanoTime() - dllibStart

      val ckksStart = System.nanoTime()
      val output2 = linear2.forward(input).toTensor[Float]
      ckksPreTime(1) += System.nanoTime() - ckksStart
      val enInput = ckks.ckksEncrypt(encryptorPtr, output2.storage().array())
      val enPre = ckks.sigmoidForward(ckksRunnerPtr, enInput)
      val pre2 = ckks.ckksDecrypt(encryptorPtr, enPre(0))
      ckksPreTime(0) += System.nanoTime() - ckksStart

      println("target dllib ckks")
      (0 until currentBs).foreach { i =>
        val dllibPre = pre.toTensor[Float].valueAt(i + 1, 1)
        val ckksPre = pre2(i)
        val t = target.valueAt(i + 1, 1)
        if (t == 0) {
          if (dllibPre <= 0.5) {
            accDllib += 1
          }
          if (ckksPre <= 0.5) {
            accCkks += 1
          }
        } else {
          if (dllibPre > 0.5) {
            accDllib += 1
          }
          if (ckksPre > 0.5) {
            accCkks += 1
          }
        }
//        println(t + " " + dllibPre + " " + ckksPre)
      }
    }
    println(s"ckksTime: " + ckksPreTime(0) / 1e9)
    println(s"dllibTime: " + dllibPreTime(0) / 1e9)
    println(s"ckks Linear Time: ${ckksPreTime(1) / 1e9}")
    println(s"dllib Linear Time: ${dllibPreTime(1) / 1e9}")
    println(s"predict correct ${accDllib} ${accCkks}")
    println(validationpairFeatureRdds.count())
  }
}

class SigmoidSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val sigmoid = Sigmoid[Float]().setName("sigmoid")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(sigmoid, input)
  }
}
