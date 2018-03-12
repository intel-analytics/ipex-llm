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

package com.intel.analytics.bigdl.dataset.datamining

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.{NumericBoolean, NumericDouble, NumericFloat, NumericInt, NumericLong, NumericShort, NumericString}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * ======RowTransformer transform a `Row` to a `Table` whose values are all `Tensor`.======
 *
 * This transformer is a container of `RowTransformSchema`s.
 * When this transformer being executed,
 * it will run `transform` methods of its `RowTransformSchema`s.
 *
 * Output of `RowTransformer` is a `Table`.
 * The keys of `Table` are Tensor.scalar(`schemaKey`)s of included `RowTransformSchema`s.
 * Correspondingly, the values of `Table` are results(`Tensor`) of `RowTransformSchema.transform`.
 *
 * @param schemas schemas of transformer, whose keys should `NOT` be duplicated
 * @param rowSize size of `Row` transformed by this transformer, default is `None`
 */
class RowTransformer(
  @transient private val schemas: Seq[RowTransformSchema],
  protected val rowSize: Option[Int] = None
) extends Transformer[Row, Table] {

  protected val schemaMap: mutable.Map[String, RowTransformSchema] = {
    val map = mutable.LinkedHashMap[String, RowTransformSchema]()
    schemas.foreach { schema =>
      require(!map.contains(schema.schemaKey),
        s"Found replicated schemeKey: ${schema.schemaKey}"
      )
      if (schema.fieldNames.isEmpty) {
        require(schema.indices.forall(i => i >= 0 && i < rowSize.getOrElse(Int.MaxValue)),
          s"At least one of indices are out of bound: ${schema.indices.mkString(",")}"
        )
      }
      map += schema.schemaKey -> schema
    }
    map
  }

  override def apply(prev: Iterator[Row]): Iterator[Table] = {
    new Iterator[Table] {
      override def hasNext: Boolean = prev.hasNext

      override def next(): Table = {
        val row = prev.next()
        val table = T()
        schemaMap.foreach { case (key, schema) =>
          val indices = schema match {
            case sch if sch.fieldNames.nonEmpty =>
              schema.fieldNames.map(row.fieldIndex)
            case sch if sch.indices.nonEmpty =>
              schema.indices
            case _ =>
              0 until row.length
          }

          val (values, fields) = indices.map(i =>
            row.get(i) -> row.schema.fields(i)
          ).unzip

          val outputKey = Tensor.scalar[String](key)
          val output = schema.transform(values, fields)

          table.update(outputKey, output)
        }
        table
      }
    }
  }

}

object RowTransformer {

  def apply(
    schemas: Seq[RowTransformSchema],
    rowSize: Int = 0
  ): RowTransformer = {
    new RowTransformer(schemas, if (rowSize > 0) Some(rowSize) else None)
  }

  /**
   * A `RowTransformer` which transform each `selected columns` to a size(1) `Tensor`.
   * The keys of output `Table` are `fieldNames` of `selected columns`.
   *
   * @param fieldNames field names of `selected columns`
   */
  def atomic(fieldNames: Seq[String]): RowTransformer = {
    val transSchemas = fieldNames.map(f => ColToTensor(f, f))
    new RowTransformer(transSchemas)
  }

  /**
   * A `RowTransformer` which transform each `selected columns` to a size(1) `Tensor`.
   * The keys of output `Table` are `indices` of `selected columns`.
   *
   * @param indices indices of `selected columns`
   * @param rowSize size of `Row` transformed by this transformer
   */
  def atomic(indices: Seq[Int], rowSize: Int): RowTransformer = {
    val transSchemas = indices.map(f => new ColToTensor(f.toString, f))
    new RowTransformer(transSchemas, Option(rowSize))
  }

  /**
   * A `RowTransformer` which concat values of `all columns` to one `Tensor`.
   * It means you will get a Table with single key-value pair after transformation.
   * The unique key is `schemaKey`. The unique value is a size(length of Row) Tensor.
   *
   * @param schemaKey key of the schema, default value is "all"
   */
  def numeric[T: ClassTag](schemaKey: String = "all"
  )(implicit ev: TensorNumeric[T]): RowTransformer = {
    new RowTransformer(Seq(ColsToNumeric[T](schemaKey)))
  }

  /**
   * A `RowTransformer` which concat values of `selected columns` to one `Tensor`.
   * It means you will get a `Table` with keys of `numericFields`.
   * Values of `Table` are `Tensor`s concatenated by `selected columns` of the keys.
   *
   * @param numericFields Map<`schemaKey`, `fieldNames of selected columns`> of numeric fields
   */
  def numeric[T: ClassTag](numericFields: Map[String, Seq[String]]
  )(implicit ev: TensorNumeric[T]): RowTransformer = {
    val transSchemas = numericFields.map { case(key, fields) => ColsToNumeric[T](key, fields) }
    new RowTransformer(transSchemas.toSeq)
  }

  /**
   * A `RowTransformer` which contains both `atomic` schemas and `numeric` schemas.
   *
   * @param atomicFields field names of `selected columns`
   * @param numericFields Map<`schemaKey`, `fieldNames of selected columns`> of numeric fields
   */
  def atomicWithNumeric[T: ClassTag](
    atomicFields: Seq[String],
    numericFields: Map[String, Seq[String]]
  )(implicit ev: TensorNumeric[T]): RowTransformer = {
    val transSchemas = mutable.ArrayBuffer[RowTransformSchema]()
    atomicFields.foreach(f => transSchemas += ColToTensor(f, f))
    numericFields.foreach { case(key, fields) =>
      transSchemas += ColsToNumeric[T](key, fields)
    }
    new RowTransformer(transSchemas)
  }

}

/**
 * A `schema` describe a transforming job which convert a `Row` to a `Table`(`Tensor`).
 */
trait RowTransformSchema extends Serializable {

  /**
   * Key of the schema, which will be the key of `Tensor` in result `Table`.
   * So, it should be `unique` in single `RowTransformer`.
   */
  val schemaKey: String

  /**
   * ======`Indices` of Selected Columns======
   * It will work on only when `fieldNames` is empty,
   * otherwise `RowTransformer` will select columns accord to `fieldNames`.
   * If both `indices` and `fieldNames` are empty,
   * `RowTransformer` will select all columns by default.
   */
  val indices: Seq[Int] = Seq.empty

  /**
   * ======`FieldNames` of Selected Columns======
   * This property will override `indices` when it is not empty.
   */
  val fieldNames: Seq[String] = Seq.empty

  /**
   * Transforming Logic of the Schema
   *
   * @param values values of selected columns
   * @param fields StructFields of selected columns
   * @return a result `Tensor`
   */
  def transform(values: Seq[Any], fields: Seq[StructField]): Tensor[NumericWildcard]

}

/**
 * A schema which specialize on transforming multiple `numeric` columns to one `Tensor`.
 * Types of `selected columns` will be identified according to their `[DataType`.
 * And type conversions will be done automatically from `DataType` to `T` if valid.
 * Currently, `DoubleType`, `FloatType`, `ShortType`, `IntegerType`, `LongType` are supported.
 *
 * @param schemaKey key of the schema
 * @param indices indices of `selected columns`
 * @param fieldNames field names of `selected columns`
 * @tparam T the type of result `Tensor`
 */
class ColsToNumeric[@specialized T: ClassTag](
  override val schemaKey: String,
  override val indices: Seq[Int] = Seq.empty,
  override val fieldNames: Seq[String] = Seq.empty
)(implicit ev: TensorNumeric[T]) extends RowTransformSchema {

  override def transform(input: Seq[Any], fields: Seq[StructField]): Tensor[NumericWildcard] = {
    val tensor = Tensor[T](input.length)
    var i = 0
    while (i < input.length) {
      val value = fields(i).dataType match {
        // TODO: support VectorUDT
        case _: DoubleType => ev.fromType(input(i).asInstanceOf[Double])
        case _: FloatType => ev.fromType(input(i).asInstanceOf[Float])
        case _: ShortType => ev.fromType(input(i).asInstanceOf[Short])
        case _: IntegerType => ev.fromType(input(i).asInstanceOf[Int])
        case _: LongType => ev.fromType(input(i).asInstanceOf[Long])
        case tpe => throw new IllegalArgumentException(s"Found unSupported DataType($tpe)!")
      }
      tensor.setValue(i + 1, value)
      i += 1
    }
    tensor.asInstanceOf[Tensor[NumericWildcard]]
  }

}

object ColsToNumeric {

  /**
   * Build a `ColsToNumeric` which transforms `all columns` of Row.
   *
   * @param schemaKey key of the schema
   * @tparam T the type of result `Tensor`
   */
  def apply[@specialized(Float, Double) T: ClassTag](schemaKey: String
  )(implicit ev: TensorNumeric[T]): ColsToNumeric[T] = {
    new ColsToNumeric[T](schemaKey)
  }

  /**
   * Build a `ColsToNumeric` which transforms `selected columns` of Row.
   *
   * @param schemaKey key of the schema
   * @param fieldNames field names of `selected columns`
   * @tparam T the type of result `Tensor`
   */
  def apply[@specialized(Float, Double) T: ClassTag](
    schemaKey: String,
    fieldNames: Seq[String]
  )(implicit ev: TensorNumeric[T]): ColsToNumeric[T] = {
    new ColsToNumeric[T](schemaKey, Seq.empty, fieldNames)
  }

}

/**
 * A schema which specialize on transforming `single column` to size(1) `Tensor`.
 * Types of `selected columns` will be identified according to their `DataType`.
 * And type conversions will be done automatically from `DataType` to `TensorDataType`.
 *
 * @param schemaKey key of the schema
 * @param index index of selected column, overridden by non empty `fieldName`
 * @param fieldName field name of selected column, default is empty
 */
class ColToTensor(
  override val schemaKey: String,
  index: Int,
  fieldName: String = ""
) extends RowTransformSchema {

  override val indices: Seq[Int] = Seq(index)

  override val fieldNames: Seq[String] = if (fieldName.isEmpty) Seq.empty else Seq(fieldName)

  override def transform(input: Seq[Any], fields: Seq[StructField]): Tensor[NumericWildcard] = {
    val (value, tpe) = input.head -> fields.head.dataType
    val tensor = tpe match {
      // TODO: support VectorUDT
      case _: BooleanType => Tensor[Boolean](1).setValue(1, value.asInstanceOf[Boolean])
      case _: DoubleType => Tensor[Double](1).setValue(1, value.asInstanceOf[Double])
      case _: FloatType => Tensor[Float](1).setValue(1, value.asInstanceOf[Float])
      case _: StringType => Tensor[String](1).setValue(1, value.asInstanceOf[String])
      case _: ShortType => Tensor[Short](1).setValue(1, value.asInstanceOf[Short])
      case _: IntegerType => Tensor[Int](1).setValue(1, value.asInstanceOf[Int])
      case _: LongType => Tensor[Long](1).setValue(1, value.asInstanceOf[Long])
      case t => throw new IllegalArgumentException(s"Found unSupported DataType($t)!")
    }
    tensor.asInstanceOf[Tensor[NumericWildcard]]
  }

}

object ColToTensor {
  /**
   * Build a `ColsToTensor` according to `fieldName`
   *
   * @param schemaKey key of the schema
   * @param fieldName field name of selected column
   */
  def apply(schemaKey: String, fieldName: String): ColToTensor = {
    new ColToTensor(schemaKey, -1, fieldName)
  }
}
