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
package com.intel.analytics.bigdl.dlframes

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature, MatToTensor}
import org.apache.spark.ml.DLTransformerBase
import org.apache.spark.ml.adapter.{HasInputCol, HasOutputCol, SchemaUtils}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

/**
 * Provides DataFrame-based API for image pre-processing and feature transformation.
 * DLImageTransformer follows the Spark Transformer API pattern and can be used as one stage
 * in Spark ML pipeline.
 *
 * The input column can be either DLImageSchema.byteSchema or DLImageSchema.floatSchema. If
 * using DLImageReader, the default format is DLImageSchema.byteSchema
 * The output column is always DLImageSchema.floatSchema.
 *
 * @param transformer Single or a sequence of BigDL FeatureTransformers to be used. E.g.
 *                    Resize(256, 256) -> CenterCrop(224, 224) ->
 *                    ChannelNormalize(123, 117, 104, 1, 1, 1) -> MatToTensor()
 */
class DLImageTransformer (
    val transformer: Transformer[ImageFeature, ImageFeature],
    override val uid: String)
  extends DLTransformerBase with HasInputCol with HasOutputCol {

  def this(transformer: FeatureTransformer) =
    this(transformer, Identifiable.randomUID("DLImageTransformer"))

  setDefault(inputCol -> "image")
  def setInputCol(value: String): this.type = set(inputCol, value)

  setDefault(outputCol -> "output")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def validateInputType(inputType: DataType): Unit = {
    val validTypes = Array(DLImageSchema.floatSchema, DLImageSchema.byteSchema)

    require(validTypes.exists(t => SchemaUtils.sameType(inputType, t)),
      s"Bad input type: $inputType. Requires ${validTypes.mkString(", ")}")
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    validateInputType(inputType)
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }

    val outputFields = schema.fields :+
      StructField($(outputCol), DLImageSchema.floatSchema, nullable = false)
    StructType(outputFields)
  }

  protected override def internalTransform(dataFrame: DataFrame): DataFrame = {
    transformSchema(dataFrame.schema, logging = true)
    val sc = dataFrame.sqlContext.sparkContext
    val localTransformer = this.transformer
    val transformerBC = sc.broadcast(localTransformer)
    val toTensorBC = sc.broadcast(MatToTensor[Float](shareBuffer = true))

    val inputColIndex = dataFrame.schema.fieldIndex($(inputCol))
    val resultRDD = dataFrame.rdd.mapPartitions { rowIter =>
      val localTransformer = transformerBC.value.cloneTransformer()
      val toTensorTransformer = toTensorBC.value.cloneTransformer().asInstanceOf[MatToTensor[Float]]
      rowIter.map { row =>
        val imf = DLImageSchema.row2IMF(row.getAs[Row](inputColIndex))
        val output = localTransformer.apply(Iterator(imf)).toArray.head
        if (!output.contains(ImageFeature.imageTensor)) {
          toTensorTransformer.transform(output)
        }
        Row.fromSeq(row.toSeq ++ Seq(DLImageSchema.imf2Row(output)))
      }
    }

    val resultSchema = transformSchema(dataFrame.schema)
    dataFrame.sqlContext.createDataFrame(resultRDD, resultSchema)
  }
}
