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
package org.apache.spark.ml

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasLabelCol
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
 * Handle different Vector types in Spark 1.5/1.6 and Spark 2.0+.
 * Support both ML Vector and MLlib Vector for Spark 2.0+.
 */
trait VectorCompatibility {

  val validVectorTypes = Seq(new VectorUDT, new org.apache.spark.mllib.linalg.VectorUDT)

  def getVectorSeq(row: Row, colType: DataType, index: Int): Seq[AnyVal] = {
    if (colType == new VectorUDT) {
      row.getAs[Vector](index).toArray.toSeq
    } else if (colType == new org.apache.spark.mllib.linalg.VectorUDT) {
      row.getAs[org.apache.spark.mllib.linalg.Vector](index).toArray.toSeq
    } else {
      throw new IllegalArgumentException(
        s"$colType is not a supported vector type.")
    }
  }
}


/**
 *A wrapper from org.apache.spark.ml.Estimator
 * Extends MLEstimator and override process to gain compatibility with
 * both spark 1.5 and spark 2.0.
 */
abstract class DLEstimatorBase[Learner <: DLEstimatorBase[Learner, M],
    M <: DLTransformerBase[M]]
  extends Estimator[M] with HasLabelCol {

  protected def internalFit(dataFrame: DataFrame): M

  override def fit(dataset: Dataset[_]): M = {
    transformSchema(dataset.schema, logging = true)
    internalFit(dataset.toDF())
  }

  override def copy(extra: ParamMap): Learner = defaultCopy(extra)

}



