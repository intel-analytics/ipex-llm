
package org.apache.spark.ml

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class MlTransform(override val uid: String) extends Transformer with HasInputCol with HasOutputCol with DataParams{

  def this() = this(Identifiable.randomUID("MlTransform"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def predict(features: Any): Float = {
    val model = this.extractParamMap().get(modelTrain).getOrElse(null)

    val result = model.forward(features.asInstanceOf[Tensor[Float]]).toTensor[Float]
    require(result.dim() == 1)

    if (result.size(1) == 1) {
      result(Array(1))
    } else {
      val maxVal = result.max(1)._2
      require(maxVal.nDimension() == 1 && maxVal.size(1) == 1)
      maxVal(Array(1)) - 1
    }
  }

  override def transform(dataset: DataFrame): DataFrame = {
    val predictUDF = udf {
      (features: Any) => {
        predict(features)
      }
    }
    dataset.withColumn($(outputCol), predictUDF(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def copy(extra: ParamMap): MlTransform = defaultCopy(extra)
}

trait DataParams extends Params {
  final val modelTrain = new Param[Module[Float]](this, "module factory", "neural network model")
  final val batchNum = new IntParam(this, "batch number", "how many batches on one partition in one iteration")

  final def getModel = $(modelTrain)
  final def getBatchNum = $(batchNum)
}

case object AnyType extends AnyType
class AnyType extends DataType {
  override def defaultSize: Int = 1
  override def asNullable: AnyType = this
}
