/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.parameters

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.optim.DistributedOptimizer.CachedModel
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

trait ParameterManager[T] extends Serializable {

  def sync(parameters: RDD[Tensor[T]]): RDD[Tensor[T]]

  def syncAndinitG(parameters: RDD[Tensor[T]], mulitleModels: RDD[Array[CachedModel[T]]]):
    RDD[Tensor[T]]  = ???

  def sumAndUpdate(parameters: RDD[Tensor[T]],
    update: (Tensor[T], Tensor[T], Table) => Unit): Unit

  def getParameter(): Tensor[T]

  def getState(): Table
}

abstract trait Parameter[T] extends Serializable {

  def copyTo(srcOffset: Int, tensor: Tensor[T], tgtOffset: Int, length: Int): Unit

  def copyTo(tensor: Tensor[T]): Unit

  def bytes(offset: Int, length: Int): ByteBuffer

  def bytes(): ByteBuffer

  def add(data: ByteBuffer, offset: Int, length: Int): this.type

  def add(data: ByteBuffer): this.type

  def parAdd(data: ByteBuffer, offset: Int, length: Int): this.type

  def parAdd(data: ByteBuffer): this.type

  def copyFrom(offset: Int, src: Tensor[T], srcOffset: Int, length: Int): this.type

  def copyFrom(tensor: Tensor[T]): this.type
}

object Parameter {
  def apply[T: ClassTag](t: Tensor[T]): Parameter[T] = new FP16Parameter(t)

  def apply[T: ClassTag](b: ByteBuffer): Parameter[T] = new FP16Parameter(b)
}
