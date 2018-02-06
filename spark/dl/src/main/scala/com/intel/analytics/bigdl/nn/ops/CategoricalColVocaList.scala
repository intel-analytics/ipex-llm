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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.hashing.MurmurHash3

/**
 * CategoricalColVocaList operation having an vocabulary mapping feature string to Integer ID
 *
 * By default, out-of-vocabulary values are ignored.
 * Use either (but not both) of num_oov_buckets and default_value
 * to specify how to include out-of-vocabulary values.
 *
 * The num_oov_buckets is Non-negative Integer.
 *
 * All out-of-vocabulary inputs will be assigned IDs in the range
 * [len(vocabulary_list), len(vocabulary_list)+num_oov_buckets) based on a hash of the input value
 *
 * A positive num_oov_buckets can not be specified with default_value.
 *
 * the input Tensor[String] can be 1-D or 2-D Tensor.
 *
 * The Operation support the feature column with single-value or multi-value
 *
 * The missing values in input Tensor can be represented by -1 for int and '''' for string
 *
 * @param vocaList An vocabulary with the length more than or equal to 1.
 * @param strDelimiter The delimiter of feature string, default: ",".
 * @param defaultValue The integer ID value for out-of-vocabulary feature values, default: -1.
 * @param numOovBuckets the number of out-of-vocabulary buckets, default: 2.
 * @tparam T Numeric type. Parameter tensor numeric type. Only support float/double now
 */

class CategoricalColVocaList[T: ClassTag](
  val vocaList: Array[String],
  val strDelimiter: String = ",",
  val defaultValue: Int = -1,
  val numOovBuckets: Int = 0
) (implicit ev: TensorNumeric[T])
  extends Operation[Tensor[String], Tensor[Int], T]{

  private val vocaLen = vocaList.length
  private val vocaMap = vocaList.zipWithIndex.toMap

  require(numOovBuckets >= 0,
    "numOovBuckets is a negative integer")
  require(!(defaultValue != -1 && numOovBuckets != 0),
    "defaultValue and numOovBuckets are both specified")
  require(vocaLen > 0,
    "the vocabulary list is empty")
  require(vocaLen == vocaMap.size,
    "the vocabulary list contains duplicate keys")

  output = Tensor[Int]()

  override def updateOutput(input: Tensor[String]): Tensor[Int] = {

    val inputSize = input.size()
    val rows = inputSize(0)
    if (inputSize.length==1) {
      input.resize(rows, 1)
    }
    val cols = if (numOovBuckets==0) vocaLen + 1 else vocaLen + numOovBuckets
    val shape = Array(rows, cols)
    val indices0 = new ArrayBuffer[Int]()
    val indices1 = new ArrayBuffer[Int]()
    val values = new ArrayBuffer[Int]()

    var i = 1
    while (i <= rows) {
      val feaStrArr = input.valueAt(i, 1).split(strDelimiter)
      var j = 0
      while (j < feaStrArr.length) {
        val mapVal = numOovBuckets==0 match {
          case true =>
            vocaMap.getOrElse(feaStrArr(j), defaultValue)
          case false =>
            vocaMap.getOrElse(feaStrArr(j),
              MurmurHash3.stringHash(feaStrArr(j)) % numOovBuckets match {
                case v if v < 0 => v + numOovBuckets + vocaLen
                case v if v >= 0 => v + vocaLen
              })
        }
        indices0 += i-1
        indices1 += j
        values += mapVal
        j += 1
      }
      i += 1
    }
    val indices = Array(indices0.toArray, indices1.toArray)
    output = Tensor.sparse(indices, values.toArray, shape)
    output
  }
}

object CategoricalColVocaList {
  def apply[T: ClassTag](
    vocaList: Array[String],
    strDelimiter: String = ",",
    defaultValue: Int = -1,
    numOovBuckets: Int = 0
  ) (implicit ev: TensorNumeric[T]): CategoricalColVocaList[T]
  = new CategoricalColVocaList[T](
    vocaList = vocaList,
    strDelimiter = strDelimiter,
    defaultValue = defaultValue,
    numOovBuckets = numOovBuckets
  )
}
