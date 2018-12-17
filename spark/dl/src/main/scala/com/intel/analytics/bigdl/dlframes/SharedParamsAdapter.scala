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

package org.apache.spark.ml.adapter

import org.apache.spark.sql.types.{DataType, StructField, StructType}


trait HasPredictionCol extends org.apache.spark.ml.param.shared.HasPredictionCol

trait HasFeaturesCol extends org.apache.spark.ml.param.shared.HasFeaturesCol

trait HasInputCol extends org.apache.spark.ml.param.shared.HasInputCol

trait HasOutputCol extends org.apache.spark.ml.param.shared.HasOutputCol

object SchemaUtils {

  /**
   * Appends a new column to the input schema. This fails if the given output column already exists
   * @param schema input schema
   * @param colName new column name. If this column name is an empty string "", this method returns
   *                the input schema unchanged. This allows users to disable output columns.
   * @param dataType new column data type
   * @return new schema with the input column appended
   */
  def appendColumn(
      schema: StructType,
      colName: String,
      dataType: DataType,
      nullable: Boolean = false): StructType = {

    val colSF = StructField(colName, dataType, nullable)
    require(!schema.fieldNames.contains(colSF.name), s"Column ${colSF.name} already exists.")
    StructType(schema.fields :+ colSF)
  }

  def sameType(a: DataType, b: DataType): Boolean = a.sameType(b)

}
