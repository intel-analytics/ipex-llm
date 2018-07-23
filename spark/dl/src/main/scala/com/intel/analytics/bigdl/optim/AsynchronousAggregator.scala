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
package com.intel.analytics.bigdl.optim

import java.util.concurrent._

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.ParameterSynchronizer
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}

import scala.collection.mutable
import scala.reflect.ClassTag


class AsynchronousAggregator {
}

trait ParaAllReduce {
  def allReduce[T: ClassTag](name: String): Tensor[T]
  def broadCast[T: ClassTag](name: String, tensor: Tensor[T]): Unit
  def init[T: ClassTag](name: String): Unit
  def test(name: String): Boolean
  def ready(name: String): Unit
}

class SimulatedParaAllReduce(partitionNum: Int, cores: Int) extends ParaAllReduce {
  val executorService = Executors.newFixedThreadPool(10)
  val layerMap: mutable.HashMap[String, Int] = new mutable.HashMap[String, Int]()
  val layerReduceMap: mutable.HashMap[String, Future[Tensor[_]]]
  = new mutable.HashMap[String, Future[Tensor[_]]]()
  val layerStateMap: mutable.HashMap[String, Boolean] = new mutable.HashMap[String, Boolean]()
  val localSyncStatus = new ConcurrentHashMap[String, String]()
  override def allReduce[T: ClassTag](name: String): Tensor[T] = {
    if (!layerReduceMap.contains(name)) {
      return null
    }
    val res = layerReduceMap.get(name).get.get()
    println(s"starting to sync here")
    ParameterSynchronizer.sync(name)
    println(s"syned here")
    layerReduceMap.remove(name)
    println(s"starting to remove here")
    // BlockManagerWrapper.removeBlock(getBlockId(name))
    println(s"removed here")
  //  layerStateMap.put(name, true)
    res.asInstanceOf[Tensor[T]]
  }

  override def init[T: ClassTag](name: String): Unit = {
    layerMap.put(name, partitionNum)
    println(s"initialize for ${name}")
   // layerStateMap.put(name, true)
    ParameterSynchronizer.register(name, cores)
  }

  override def broadCast[T: ClassTag](name: String, tensor: Tensor[T]): Unit = {
    // simulate local sync
    ParameterSynchronizer.syncData(name, tensor)
    val all = ParameterSynchronizer.collect(name)
    val active = ParameterSynchronizer.reset(name)
    if (active) {
      BlockManagerWrapper.putSingle(getBlockId(name),
        tensor, StorageLevel.MEMORY_ONLY_SER)
      layerStateMap.put(name, false)
      println(s"${name} state changed to false")
      val future = executorService.submit(new AyncFetch(this, layerMap.get(name).get,
        name, tensor)).
        asInstanceOf[Future[Tensor[_]]]
      layerReduceMap.put(name, future)
    }
  }

  private def getBlockId(name: String): BlockId = {
    SparkExtension.getLocalBlockId(name)
  }

  class AyncFetch[T](allReduce: ParaAllReduce,
                     worldSize: Int, name: String, simulatedTensor: Tensor[T] = null)
    extends Callable[Tensor[T]] {
    override def call(): Tensor[T] = {
      var tensor : Tensor[T] = null
      (0 until worldSize).map(index => {
        BlockManagerWrapper.getLocalOrRemoteBytes(getBlockId(name)).get
      })
      allReduce.ready(name)
      simulatedTensor
    }
  }

  override def test(name: String): Boolean = {
    return layerStateMap.get(name).get
  }

  override def ready(name: String): Unit = {
    layerStateMap.put(name, true)
  }
}


