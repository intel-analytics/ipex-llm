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
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * A wrapper for org.apache.spark.ml.Transformer.
 * Extends MlTransformer and override process to gain compatibility with
 * both spark 1.5 and spark 2.0.
 */
abstract class MlTransformer extends Transformer{

  def process(dataset: DataFrame): DataFrame

  override def transform(dataset: Dataset[_]): DataFrame = {
    process(dataset.toDF())
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def copy(extra: ParamMap): MlTransformer = defaultCopy(extra)
}
