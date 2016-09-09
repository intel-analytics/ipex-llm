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

package com.intel.analytics.sparkdl.pipeline

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.optim.SGD
import com.intel.analytics.sparkdl.utils.T
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{NNClassifier, Pipeline}
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}
import org.scalatest.{FlatSpec, Matchers}

class NNClassifierSpec extends FlatSpec with Matchers {
  "classifier with MSE and SGD" should "get good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1 : Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2 : Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(1 to 256 * 8, 1).map{ index =>
      if (index % 2 == 0) {
        new LabeledPoint(output1, new DenseVector(input1))
      } else {
        new LabeledPoint(output2, new DenseVector(input2))
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)

    val state = T("maxIter" -> 100, "learningRate" -> 20.0)

    val classifier = new NNClassifier()
    classifier.setCriterion(new MSECriterion[Double])
    classifier.setModel(_ => mlp)
    classifier.setOptimizerType("serial")
    classifier.setOptMethod(new SGD[Double])
    classifier.setBatchSize(256)
    classifier.setBatchNum(1)
    classifier.setState(state)

    val sqlContext = new SQLContext(sc)
    val trainData = sqlContext.createDataFrame(data)

    val model = classifier.fit(trainData)

    val testData = sqlContext.createDataFrame(sc.parallelize(Array(
      new LabeledPoint(output1, new DenseVector(input1)),
      new LabeledPoint(output2, new DenseVector(input2))
    )))

    val result = model.transform(testData)
    result.select("prediction", "label").map {
      case Row(prediction : Double, label : Double) => (prediction, label)
    }.collect().map{case (prediction, label) => prediction should be(label +- 5e-2)}


    result.select("prediction", "rawPrediction").map {
      case Row(prediction : Double, rawPrediction : Vector) => (prediction, rawPrediction)
    }.collect().map{case (prediction, rawPrediction) => prediction should be(rawPrediction(1))}

    sc.stop()
  }

  "classifier with NLL and SGD" should "get good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1 : Array[Double] = Array(0, 1, 0, 1)
    val output1 = 0.0
    val input2 : Array[Double] = Array(1, 0, 1, 0)
    val output2 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(1 to 256 * 8, 1).map{ index =>
      if (index % 2 == 0) {
        new LabeledPoint(output1, new DenseVector(input1))
      } else {
        new LabeledPoint(output2, new DenseVector(input2))
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)

    val state = T("maxIter" -> 100, "learningRate" -> 20.0)

    val classifier = new NNClassifier()
    classifier.setCriterion(new ClassNLLCriterion[Double])
    classifier.setModel(_ => mlp)
    classifier.setOptimizerType("serial")
    classifier.setOptMethod(new SGD)
    classifier.setBatchSize(256)
    classifier.setBatchNum(1)
    classifier.setState(state)

    val sqlContext = new SQLContext(sc)
    val trainData = sqlContext.createDataFrame(data)

    val model = classifier.fit(trainData)

    val testData = sqlContext.createDataFrame(sc.parallelize(Array(
      new LabeledPoint(output1, new DenseVector(input1)),
      new LabeledPoint(output2, new DenseVector(input2))
    )))

    val result = model.transform(testData)
    result.select("prediction", "label").map {
      case Row(prediction : Double, label : Double) => (prediction, label)
    }.collect().map{case (prediction, label) => prediction should be(label +- 5e-2)}


    result.select("prediction", "rawPrediction").map {
      case Row(prediction : Double, rawPrediction : Vector) => (prediction, rawPrediction)
    }.collect().map{case (prediction, rawPrediction) => prediction should be(rawPrediction(1))}

    sc.stop()
  }

  "classifier" should "support gird search" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = new SparkContext("local[1]", "SerialOptimizerSpec")

    // Prepare two kinds of input and their corresponding label
    val input1 : Array[Double] = Array(0, 1.5, 0, 1.5)
    val output1 = 0.0
    val input2 : Array[Double] = Array(0, 1.2, 0, 1.2)
    val output2 = 0.0
    val input3 : Array[Double] = Array(0, 0.8, 0, 0.8)
    val output3 = 1.0
    val input4 : Array[Double] = Array(0, 0.5, 0, 0.5)
    val output4 = 1.0

    // Generate a toy training data
    val data = sc.makeRDD(1 to 256 * 8, 1).map{ index =>
      if (index % 4 == 0) {
        new LabeledPoint(output1, new DenseVector(input1))
      } else if (index % 4 == 1) {
        new LabeledPoint(output2, new DenseVector(input2))
      } else if (index % 4 == 2) {
        new LabeledPoint(output3, new DenseVector(input3))
      } else {
        new LabeledPoint(output4, new DenseVector(input4))
      }
    }

    val mlp = new Sequential[Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)
    val initW = mlp.getParameters()._1
    initW.fill(0.01)

    val classifier = new NNClassifier()
    classifier.setCriterion(new ClassNLLCriterion[Double])
    classifier.setModel(_ => mlp)
    classifier.setOptimizerType("serial")
    classifier.setOptMethod(new SGD[Double])
    classifier.setBatchSize(256)
    classifier.setBatchNum(1)

    val sqlContext = new SQLContext(sc)
    val trainData = sqlContext.createDataFrame(data)

    val pipeline = new Pipeline().setStages(Array(classifier))

    val state1 = T("learningRate" -> 0.1, "maxIter" -> 1)

    val state2 = T("learningRate" -> 0.1, "maxIter" -> 50)

    val state3 = T("learningRate" -> 0.1, "maxIter" -> 100)

    val paramGrid = new ParamGridBuilder().addGrid(classifier.state,
      Array(state1, state2, state3)).build()
    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setMetricName("areaUnderROC")

    val cv = new CrossValidator().setNumFolds(4).setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid).setEvaluator(evaluator)
    val cvModel = cv.fit(trainData)

    println(s"auc is [${cvModel.avgMetrics.mkString(",")}]")
    cvModel.avgMetrics(0) < cvModel.avgMetrics(1) should be (true)
    cvModel.avgMetrics(1) <= cvModel.avgMetrics(2) should be (true)

    sc.stop()
  }
}
