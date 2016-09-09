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

package org.apache.spark.ml

import com.intel.analytics.sparkdl.nn.{ClassNLLCriterion, Criterion, Module}
import com.intel.analytics.sparkdl.optim._
import com.intel.analytics.sparkdl.ps.{OneReduceParameterManager, ParameterManager}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.utils.{T, Table}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}

import scala.reflect.ClassTag

trait NNParams[@specialized(Float, Double) T] extends PredictorParams {

  final val model: Param[Int => Module[T]] =
    new Param(this, "module factory", "neural network model")

  final val criterion: Param[Criterion[T]] =
    new Param(this, "criterion", "criterion that evaluate the result")

  final val state: Param[Table] = new Param(this, "state", "states to train the neural network")

  final val optMethod: Param[OptimMethod[T]] =
    new Param(this, "optimize method", "optimize method")

  final val optimizerType: Param[String] =
    new Param(this, "optimizer type", "distributed optimizer type",
      ParamValidators.inArray(Array("serial", "parallel")))

  final val batchSize: IntParam =
    new IntParam(this, "batch size", "how much data in one forward/backward", ParamValidators.gt(0))

  final val batchNum: IntParam =
    new IntParam(this, "batch number", "how many batches on one partition in one iteration")

  final val modelPath: Param[String] =
    new Param(this, "model path", "where to store the model")

  final def getOptimizerType: String = $(optimizerType)

  final def getModel: Int => Module[T] = $(model)

  final def getState: Table = $(state)

  final def getOptMethod: OptimMethod[T] = $(optMethod)

  final def getCriterion: Criterion[T] = $(criterion)

  final def getBatchSize: Int = $(batchSize)

  final def getBatchNum: Int = $(batchNum)

  final def getModelPath: String = $(modelPath)
}

/**
 * This is a spark ml classifier wrapper.
 */
class NNClassifier(override val uid: String)
  extends Predictor[Vector, NNClassifier, NNClassificationModel[Double]]
    with NNParams[Double] {

  private var initState: Option[Table] = None

  def this() = this(Identifiable.randomUID("nnc"))

  def setModel(value: Int => Module[Double]): this.type = {
    set(model, value)
  }

  def setState(value: Table): this.type = {
    initState = Some(value.clone())
    set(state, value)
  }

  def setOptMethod(value: OptimMethod[Double]): this.type = set(optMethod, value)

  def setOptimizerType(value: String): this.type = set(optimizerType, value)

  def setCriterion(value: Criterion[Double]): this.type = set(criterion, value)

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  def setBatchNum(value: Int): this.type = set(batchNum, value)

  def setModelPath(value: String): this.type = set(modelPath, value)

  setDefault(modelPath -> null, batchNum -> -1, state -> T())

  override def copy(extra: ParamMap): NNClassifier = {
    val tmpState = extra.remove(state)
    val that: NNClassifier = defaultCopy(extra)

    if (tmpState.isDefined) {
      that.setState(tmpState.get.clone())
      extra.put(state, tmpState.get)
    } else {
      that.setState(initState.get.clone())
    }
    that
  }

  override protected def train(data: DataFrame): NNClassificationModel[Double] = {
    val featureSize = data.first().getAs[Vector]("features").size
    val trainData = data.select($(labelCol), $(featuresCol)).map {
      case Row(label: Double, features: Vector) =>
        LabeledPoint(label, features.toDense)
    }
    val module = $(model)(featureSize).cloneModule()
    val weights = module.getParameters()._1
    val metrics = new Metrics
    val offset = if ($(criterion).isInstanceOf[ClassNLLCriterion[Double]]) 1 else 0
    val dataSets = new ShuffleBatchDataSet[LabeledPoint, Double](trainData, toTensor(offset) _,
      $(batchSize), $(batchSize), $(batchNum))
    val pm = new OneReduceParameterManager[Double](weights, dataSets.partitions(), metrics)
    val optimizer = getOptimizer(module, featureSize, dataSets, pm, metrics)
    optimizer.setPath($(modelPath))

    optimizer.optimize()

    new NNClassificationModel(uid, optimizer.module)
  }

  private def getOptimizer(module: Module[Double], featureSize: Int,
    dataset: DataSet[_, Double] with HasEpoch, pm: ParameterManager[Double],
    metrics: Metrics): Optimizer[Double] = {
    val epoch = $(state)[Int]("maxIter")
    $(optimizerType) match {
      case "serial" =>
        new GradAggEpochOptimizer[Double](module, $(criterion), $(optMethod), pm, dataset,
          metrics, $(state)).setMaxEpoch(epoch)
      case "parallel" =>
        new WeightAvgEpochOptimizer[Double](module, $(criterion), $(optMethod), pm, dataset,
          metrics, $(state)).setMaxEpoch(epoch)
      case _ =>
        throw new IllegalArgumentException
    }
  }

  private def toTensor(offset: Int)(inputs: Seq[LabeledPoint], input: Tensor[Double],
    target: Tensor[Double]): (Tensor[Double], Tensor[Double]) = {
    val size = inputs.size
    require(size > 0)

    val featureSize = inputs(0).features.size
    val shape = inputs(0).features match {
      case _: DenseVector | _: SparseVector =>
        Array(featureSize)
      case _ =>
        throw new IllegalArgumentException
    }
    input.resize(Array(size) ++ shape)
    target.resize(Array(size))
    var i = 0
    while (i < size) {
      target.setValue(i + 1, (inputs(i).label + offset))
      inputs(i).features match {
        case _: DenseVector =>
          System.arraycopy(inputs(i).features.toArray, 0, input.storage().array(),
            i * featureSize, featureSize)
        case _: SparseVector =>
          var j = 0
          while (j < featureSize) {
            input.setValue(i + 1, j + 1, inputs(i).features(j))
            j += 1
          }
        case _ =>
          throw new IllegalArgumentException
      }
      i += 1
    }

    (input, target)
  }
}

class NNClassificationModel[@specialized(Float, Double) T: ClassTag](
  override val uid: String,
  val module: Module[T])(implicit ev: TensorNumeric[T])
  extends PredictionModel[Vector, NNClassificationModel[T]] with HasRawPredictionCol
    with Serializable {

  override protected def predict(features: Vector): Double = {
    var result: Tensor[T] = null
    if (T.isInstanceOf[Double]) {
      result = module.forward(torch.Tensor(torch.storage(features.toArray.asInstanceOf[Array[T]])))
    } else {
      val array: Array[T] = new Array[T](features.toArray.length)
      var index = 0
      features.toArray.foreach { x =>
        array(index) = ev.fromType[Double](x)
        index += 1
      }
      result = module.forward(torch.Tensor(torch.storage(array)))
    }

    require(result.nDimension() == 1)
    if (result.size(1) == 1) {
      ev.toType[Double](result(Array(1)))
    } else {
      val max = result.max(1)._2
      require(max.nDimension() == 1 && max.size(1) == 1)
      ev.toType[Double](max(Array(1))) - 1
    }
  }

  protected override def transformImpl(dataset: DataFrame): DataFrame = {
    val predictUDF = udf { (features: Any) =>
      predict(features.asInstanceOf[Vector])
    }

    val toVector = udf { (result: Double) =>
      new DenseVector(Array(0.0, result))
    }

    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
      .withColumn($(rawPredictionCol), toVector(col($(predictionCol))))
  }

  override def copy(extra: ParamMap): NNClassificationModel[T] = {
    copyValues(new NNClassificationModel(uid, module), extra)
  }
}
