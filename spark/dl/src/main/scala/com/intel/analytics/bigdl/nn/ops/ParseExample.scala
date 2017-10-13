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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.{T, Table}
import com.google.protobuf.ByteString
import org.tensorflow.example.{Example, Feature}
import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

class ParseExample[T: ClassTag](nDense: Int,
                                tDense: Seq[TensorDataType],
                                denseShape: Seq[Array[Int]])
     (implicit ev: TensorNumeric[T])
  extends Operation[Table, Table, T] {

  type StringType = ByteString

  override def updateOutput(input: Table): Table = {
    require(input[Tensor[StringType]](1).size(1) == 1, "only support one example at a time")
    val serialized = input[Tensor[StringType]](1).valueAt(1)
    val denseKeys = Range(3, 3 + nDense).map(index => input(index).asInstanceOf[Tensor[StringType]])
      .map(_.value().toStringUtf8)
    val denseDefault = Range(3 + nDense, 3 + 2 * nDense)
      .map(index => input(index).asInstanceOf[Tensor[_]])


    val example = Example.parseFrom(serialized)

    val featureMap = example.getFeatures.getFeatureMap

    val outputs = denseDefault
      .zip(denseKeys)
      .zip(tDense).zip(denseShape).map { case (((default, key), tensorType), shape) =>
        if (featureMap.containsKey(key)) {
          val feature = featureMap.get(key)
          getTensorFromFeature(feature, tensorType, shape)
        } else {
         default
        }
    }

    for (elem <- outputs) {
      elem.asInstanceOf[Tensor[NumericWildcard]].addSingletonDimension()
      output.insert(elem)
    }
    output
  }

  private def getTensorFromFeature(feature: Feature,
                                   tensorType: TensorDataType,
                                   tensorShape: Array[Int]): Tensor[_] = {
    tensorType match {
      case LongType =>
        val values = feature.getInt64List.getValueList.asScala.map(_.longValue()).toArray
        Tensor(values, tensorShape)
      case FloatType =>
        val values = feature.getFloatList.getValueList.asScala.map(_.floatValue()).toArray
        Tensor(values, tensorShape)
      case StringType =>
        val values = feature.getBytesList.getValueList
          .asScala.toArray.asInstanceOf[Array[ByteString]]
        Tensor(values, tensorShape)
    }
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    throw new UnsupportedOperationException("no backward on ParseExample")
  }
}
