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
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
/**
 *A wrapper from org.apache.spark.ml.Estimator
 * Extends MLEstimator and override process to gain compatibility with
 * both spark 1.5 and spark 2.0.
 */
abstract class MLEstimator extends Estimator[MlTransformer]{
  def process(dataset: DataFrame): MlTransformer
  override def fit(dataset : org.apache.spark.sql.DataFrame) : MlTransformer = {
    process(dataset)
  }
  override def transformSchema(schema: StructType): StructType = schema
  override def copy(extra: ParamMap): MLEstimator = defaultCopy(extra)
}
