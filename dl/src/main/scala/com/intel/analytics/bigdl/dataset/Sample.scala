/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Storage, Tensor}

import scala.reflect.ClassTag

/**
 * Sample, bundling input and target
 *
 * @param featureTensor
 * @param labelTensor
 * @tparam T
 */

class Sample[T: ClassTag] (
   protected var featureTensor: Tensor[T],
   protected var labelTensor: Tensor[T])
   (implicit ev: TensorNumeric[T]) extends Serializable {

 def this()(implicit ev: TensorNumeric[T]) = this(Tensor[T](), Tensor[T]())

 def set(featureData: Array[T],
          labelData: Array[T],
          featureSize: Array[Int],
          labelSize: Array[Int]): Sample[T] = {
   featureTensor.set(Storage[T](featureData), 1, featureSize)
   labelTensor.set(Storage[T](labelData), 1, labelSize)
   this
 }

 def extractFeature(storage: Array[T], offset: Int, length: Int): Unit = {
   require(offset + length <= storage.length, "index out of boundary")
   require(length <= featureTensor.storage().array().length, "length too long for feature")
   ev.getType() match {
     case DoubleType => Array.copy(featureTensor.storage().array
         .asInstanceOf[Array[Double]],
       featureTensor.storageOffset() - 1, storage
         .asInstanceOf[Array[Double]], offset, length)
     case FloatType => System.arraycopy(featureTensor.storage().array
       .asInstanceOf[Array[Float]],
       featureTensor.storageOffset() - 1, storage
       .asInstanceOf[Array[Float]], offset, length)
     case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
   }
 }

 def extractLabel(storage: Array[T], offset: Int, length: Int): Unit = {
   require(offset + length <= storage.length, "index out of boundary")
   require(length <= labelTensor.storage().array().length, "length too long for label")
   ev.getType() match {
     case DoubleType => Array.copy(labelTensor.storage().array
       .asInstanceOf[Array[Double]],
       labelTensor.storageOffset() - 1, storage
       .asInstanceOf[Array[Double]], offset, length)
     case FloatType => Array.copy(labelTensor.storage().array
       .asInstanceOf[Array[Float]],
       labelTensor.storageOffset() - 1, storage
       .asInstanceOf[Array[Float]], offset, length)
     case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
   }
 }

 def copy(other: Sample[T]): Sample[T] = {
   this.featureTensor.resizeAs(other.featureTensor).copy(other.featureTensor)
   this.labelTensor.resizeAs(other.labelTensor).copy(other.labelTensor)
   this
 }

 def feature(): Tensor[T] = featureTensor

 def label(): Tensor[T] = labelTensor

 override def clone(): Sample[T] = {
   new Sample[T]().copy(this)
 }
}

object Sample {
  def apply[@specialized(Float, Double) T: ClassTag]
  (featureTensor: Tensor[T], labelTensor: Tensor[T])
  (implicit ev: TensorNumeric[T]) : Sample[T] = {
    new Sample[T](featureTensor, labelTensor)
  }
  def apply[@specialized(Float, Double) T: ClassTag]()
  (implicit ev: TensorNumeric[T]) : Sample[T] = {
    new Sample[T]()
  }
}
