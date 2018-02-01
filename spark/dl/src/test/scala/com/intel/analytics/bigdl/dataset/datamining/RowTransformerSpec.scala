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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.NumericWildcard
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types._
import org.scalatest.{FlatSpec, Matchers}

class RowTransformerSpec extends FlatSpec with Matchers {

  private val testRow = {
    new GenericRowWithSchema(Array(1, 2L, 3.3, 4.4f, "aa", false, 77.toShort),
      StructType(Seq(StructField("int", IntegerType),
        StructField("long", LongType),
        StructField("double", DoubleType),
        StructField("float", FloatType),
        StructField("str", StringType),
        StructField("bool", BooleanType),
        StructField("short", ShortType)))
    )
  }

  private val numericRow = {
    new GenericRowWithSchema(Array(1, 1.1, 1.2f),
      StructType(Seq(StructField("int", IntegerType),
        StructField("double", DoubleType),
        StructField("float", FloatType)))
    )
  }

  private val sInt = Tensor.scalar[String]("int")
  private val sLong = Tensor.scalar[String]("long")
  private val sFloat = Tensor.scalar[String]("float")
  private val sDouble = Tensor.scalar[String]("double")
  private val sBool = Tensor.scalar[String]("bool")
  private val sStr = Tensor.scalar[String]("str")
  private val sShort = Tensor.scalar[String]("short")
  private val sWrap = (s: String) => Tensor.scalar[String](s)

  "ColToTensor" should "deal with different DataTypes correctly" in {
    ColToTensor("str", "str")
      .transform(Seq("test123"), Seq(StructField("str", StringType))
      ).toArray() shouldEqual Array("test123")
    ColToTensor("int", "int")
      .transform(Seq(1), Seq(StructField("int", IntegerType))
      ).toArray() shouldEqual Array(1)
    ColToTensor("long", "long")
      .transform(Seq(1L), Seq(StructField("long", LongType))
      ).toArray() shouldEqual Array(1L)
    ColToTensor("double", "double")
      .transform(Seq(.1), Seq(StructField("double", DoubleType))
      ).toArray() shouldEqual Array(.1)
    ColToTensor("float", "float")
      .transform(Seq(.12f), Seq(StructField("float", FloatType))
      ).toArray() shouldEqual Array(.12f)
    ColToTensor("bool", "bool")
      .transform(Seq(false), Seq(StructField("bool", BooleanType))
      ).toArray() shouldEqual Array(false)
    ColToTensor("short", "short")
      .transform(Seq(1.toShort), Seq(StructField("short", ShortType))
      ).toArray() shouldEqual Array(1.toShort)
  }

  private def mkStructFields(num: Int, dataType: DataType): Seq[StructField] = {
    (1 to num).map(i => StructField(i.toString, dataType))
  }

  "ColsToNumeric" should "deal with different DataTypes correctly" in {
    var tFloat = ColsToNumeric[Float]("int", Seq("1", "2", "3"))
      .transform(Seq(1, 2, 3), mkStructFields(3, IntegerType)
      ).asInstanceOf[Tensor[Float]]
    tFloat.storage().array() shouldEqual Array(1f, 2f, 3f)
    tFloat = ColsToNumeric[Float]("long", Seq("1", "2", "3"))
      .transform(Seq(1L, 2L, 3L), mkStructFields(3, LongType)
      ).asInstanceOf[Tensor[Float]]
    tFloat.storage().array() shouldEqual Array(1f, 2f, 3f)
    tFloat = ColsToNumeric[Float]("double", Seq("1", "2", "3"))
      .transform(Seq(1.1, 2.2, 3.3), mkStructFields(3, DoubleType)
      ).asInstanceOf[Tensor[Float]]
    tFloat.storage().array() shouldEqual Array(1.1f, 2.2f, 3.3f)
    var tDouble = ColsToNumeric[Double]("float", Seq("1", "2", "3"))
      .transform(Seq(1f, 2f, 3f), mkStructFields(3, FloatType)
      ).asInstanceOf[Tensor[Double]]
    tDouble.storage().array() shouldEqual Array(1, 2, 3)
    tDouble = ColsToNumeric[Double]("short", Seq("1", "2", "3"))
      .transform(Seq(1.toShort, 2.toShort, 3.toShort), mkStructFields(3, ShortType)
      ).asInstanceOf[Tensor[Double]]
    tDouble.storage().array() shouldEqual Array(1, 2, 3)
    intercept[Exception] {
      ColsToNumeric[Double]("bool", Seq("1", "2", "3"))
        .transform(Seq(false, true, false), mkStructFields(3, BooleanType))
    }
    intercept[Exception] {
      ColsToNumeric[Double]("str", Seq("1", "2", "3"))
        .transform(Seq("1", "2", "3"), mkStructFields(3, StringType))
    }
  }

  "RowTransformer" should "deal with atomic schema correctly" in {
    var transformer = RowTransformer.atomic(
      Seq("int", "long", "float", "double", "str", "bool", "short"))
    var table = transformer(Iterator.single(testRow)).next()
    table.get[Tensor[Int]](sInt).get.size() shouldEqual Array(1)
    table.get[Tensor[Int]](sInt).get.valueAt(1) shouldEqual 1
    table.get[Tensor[Long]](sLong).get.valueAt(1) shouldEqual 2L
    table.get[Tensor[Float]](sFloat).get.valueAt(1) shouldEqual 4.4f
    table.get[Tensor[Double]](sDouble).get.valueAt(1) shouldEqual 3.3
    table.get[Tensor[String]](sStr).get.valueAt(1) shouldEqual "aa"
    table.get[Tensor[Boolean]](sBool).get.valueAt(1) shouldEqual false
    table.get[Tensor[Short]](sShort).get.valueAt(1) shouldEqual 77.toShort
    transformer = RowTransformer.atomic(Seq(1, 3, 5), 7)
    table = transformer(Iterator.single(testRow)).next()
    table.get[Tensor[Long]](sWrap("1")).get.valueAt(1) shouldEqual 2L
    table.get[Tensor[Float]](sWrap("3")).get.valueAt(1) shouldEqual 4.4f
    table.get[Tensor[Boolean]](sWrap("5")).get.valueAt(1) shouldEqual false
    intercept[Exception] {
      RowTransformer.atomic(Seq(5, 7), 7)
    }
    transformer = RowTransformer.atomic(Seq("something"))
    val iter = transformer(Iterator.single(testRow))
    intercept[Exception] {
      iter.next()
    }
  }

  "RowTransformer" should "deal with numeric schema correctly" in {
    val numericFields = Map(
      "allNum" -> Seq("int", "short", "float", "double", "long"),
      "dupNum" -> Seq("float", "float", "float", "int", "int", "int")
    )
    var transformer = RowTransformer.numeric[Float](numericFields)
    var table = transformer(Iterator.single(testRow)).next()
    var tensor = table.get[Tensor[Float]](sWrap("allNum")).get
    tensor.size shouldEqual Array(5)
    tensor.storage().array() shouldEqual Array(1, 77, 4.4, 3.3, 2).map(_.toFloat)
    tensor = table.get[Tensor[Float]](sWrap("dupNum")).get
    tensor.size shouldEqual Array(6)
    tensor.storage().array() shouldEqual Array(4.4, 4.4, 4.4, 1, 1, 1).map(_.toFloat)

    transformer = RowTransformer.numeric[Float]()
    table = transformer(Iterator.single(numericRow)).next()
    tensor = table.get[Tensor[Float]](sWrap("all")).get
    tensor.size() shouldEqual Array(3)
    tensor.toArray() shouldEqual Array(1f, 1.1f, 1.2f)
  }

  "RowTransformer" should "deal with mixed schema correctly" in {
    val transformer = RowTransformer.atomicWithNumeric[Float](
      Seq("str", "bool"),
      Map("num" -> Seq("int", "long", "double", "float", "short")))
    val table = transformer(Iterator.single(testRow)).next()
    table.get[Tensor[String]](sStr).get.toArray() shouldEqual Array("aa")
    table.get[Tensor[Boolean]](sBool).get.toArray() shouldEqual Array(false)
    table.get[Tensor[Float]](sWrap("num")).get.toArray() shouldEqual Array(
      1, 2, 3.3, 4.4, 77).map(_.toFloat)
  }

  "RowTransformer" should "work correctly with user-defined RowTransfromSchema" in {
    val transformer = RowTransformer(Seq(new BruteForceHash(),
      ColToTensor("str", "str"), ColToTensor("long", "long"))
    )
    val table = transformer(Iterator.single(testRow)).next()
    val tensor = table.get[Tensor[Int]](sWrap("hash")).get
    tensor.size() shouldEqual Array(7)
    tensor.valueAt(2) shouldEqual table.get[Tensor[Long]](sLong
    ).get.toArray().head.toString.hashCode()
    tensor.valueAt(5) shouldEqual table.get[Tensor[String]](sStr
    ).get.toArray().head.toString.hashCode()
  }

  "Cloned RowTransformer" should "work correctly" in {
    val transformer = RowTransformer(Seq(ColToTensor("str", "str"),
      ColsToNumeric[Double]("num", Seq("int", "long")))
    )
    val table = transformer(Iterator.single(testRow)).next()
    val cloned = transformer.cloneTransformer()
    val tableCloned = cloned(Iterator.single(testRow)).next()
    table shouldEqual tableCloned
  }

  class BruteForceHash extends RowTransformSchema {
    override val schemaKey: String = "hash"

    override def transform(values: Seq[Any], fields: Seq[StructField]): Tensor[NumericWildcard] = {
      Tensor[Int](values.map(_.toString.hashCode).toArray, Array(values.length)
      ).asInstanceOf[Tensor[NumericWildcard]]
    }
  }

}
