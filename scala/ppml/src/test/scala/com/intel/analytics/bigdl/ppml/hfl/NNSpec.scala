package com.intel.analytics.bigdl.ppml.hfl

import com.intel.analytics.bigdl.ppml.{FLContext, FLServer}
import com.intel.analytics.bigdl.ppml.algorithms.hfl.LogisticRegression
import com.intel.analytics.bigdl.ppml.example.LogManager
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class MockAnotherParty(algorithm: String, clientID: String = "mock") extends Thread {

  override def run(): Unit = {
    algorithm match {
      case "logistic_regression" => runLogisticRegression()
      case _ => throw new NotImplementedError()
    }
  }
  def runLogisticRegression(): Unit = {
    val spark = FLContext.getSparkSession()
    import spark.implicits._
    val df = spark.read.option("header", "true")
      .csv(this.getClass.getClassLoader.getResource("diabetes-test.csv").getPath)
    val lr = new LogisticRegression(df.columns.size - 1)
    lr.fit(df, valData = df)
  }
}
class NNSpec extends FlatSpec with Matchers with BeforeAndAfter with LogManager {
  "Logistic Regression" should "work" in {
    val flServer = new FLServer()
    flServer.build()
    flServer.start()
    val spark = FLContext.getSparkSession()
    import spark.implicits._
    val trainDf = spark.read.option("header", "true")
      .csv(this.getClass.getClassLoader.getResource("diabetes-test.csv").getPath)
    val testDf = trainDf.drop("Outcome")
    trainDf.show()
    FLContext.initFLContext()
    val lr = new LogisticRegression(trainDf.columns.size - 1)
    lr.fit(trainDf, valData = trainDf)
    lr.evaluate(trainDf)
    lr.predict(testDf)
  }
  "Linear Regression" should "work" in {

  }

}
