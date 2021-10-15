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

/**
 * A wrapper for org.apache.spark.ml.Transformer.
 * Extends MlTransformer and override process to gain compatibility with
 * both spark 1.5 and spark 2.0.
 */
abstract class DLTransformerBase[M <: DLTransformerBase[M]]
  extends Model[M] {

  protected def internalTransform(dataFrame: DataFrame): DataFrame

  override def transform(dataFrame: DataFrame): DataFrame = {
    transformSchema(dataFrame.schema, logging = true)
    internalTransform(dataFrame)
  }

  override def copy(extra: ParamMap): M = defaultCopy(extra)
}
