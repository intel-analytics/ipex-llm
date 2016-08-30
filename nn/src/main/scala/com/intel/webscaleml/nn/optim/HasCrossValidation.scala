package com.intel.webscaleml.nn.optim

import com.intel.webscaleml.nn.nn.Module
import com.intel.webscaleml.nn.optim.Optimizer.CachedModel
import com.intel.webscaleml.nn.tensor.Tensor
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

trait HasCrossValidation[@specialized(Float, Double) T] extends Serializable with Logging{
  private var testInterval : Int = 1

  def setTestInterval(testInterval : Int) : this.type = {
    require(testInterval > 0)
    this.testInterval = testInterval
    this
  }

  val models : RDD[CachedModel[T]]

  private var testDataSet : Option[DataSet[_, T]] = None

  def setTestDataSet(testDataSet : DataSet[_, T]) : this.type = {
    this.testDataSet = Some(testDataSet)
    this
  }

  private var evaluation : Option[(Tensor[T], Tensor[T]) => (Int, Int)] = None

  def setEvaluation(evaluation : (Tensor[T], Tensor[T]) => (Int, Int)) : this.type = {
    this.evaluation = Some(evaluation)
    this
  }

  def test(module : Module[T], iter : Int, wallClockNanoTime : Option[Long] = None) : Option[Double] = {
    if(evaluation.isDefined && testDataSet.isDefined && iter % testInterval == 0) {
      val evaluationBroadcast = testDataSet.get.getSparkContext().broadcast(evaluation.get)
      val (correctSum, totalSum) = testDataSet.get.fetchAll().coalesce(models.partitions.length, false)
        .zipPartitions(models)((data, cacheModelIter) => {
          val localModel = cacheModelIter.next().model
          val localEvaluation = evaluationBroadcast.value
          Iterator.single(data.foldLeft((0, 0))((count, t) => {
            val result = localEvaluation(localModel.forward(t._1), t._2)
            (count._1 + result._1, count._2 + result._2)
          }))
        }).reduce((a, b) => (a._1 + b._1, a._2 + b._2))

      val accuracy = correctSum.toDouble / totalSum
      if(wallClockNanoTime.isDefined) {
        logInfo(s"[Wall Clock ${wallClockNanoTime.get.toDouble / 1e9}s}] correct is $correctSum total is $totalSum")
        logInfo(s"[Wall Clock ${wallClockNanoTime.get.toDouble / 1e9}s}] accuracy is $accuracy")
      } else {
        logInfo(s"correct is $correctSum total is $totalSum")
        logInfo(s"cross validation result is $accuracy")
      }
      Some(accuracy)
    } else {
      None
    }
  }
}
