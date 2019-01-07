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

package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.{SparkContext, TaskContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class DistributedSynchronizerSpec extends FlatSpec with Matchers with BeforeAndAfter {

  var sc: SparkContext = null

  before {
    val conf = Engine.createSparkConf().setAppName("test synchronizer").setMaster("local[4]")
      .set("spark.rpc.message.maxSize", "200")
    sc = new SparkContext(conf)
    Engine.init
  }

  "DistributedSynchronizer" should "work properly" in {
    val partition = 4
    val cores = 4
    val res = sc.parallelize((0 until partition), partition).mapPartitions(p => {
      Engine.setNodeAndCore(partition, cores)
      val partitionID = TaskContext.getPartitionId
      val sync = new BlockManagerParameterSynchronizer[Float](partitionID, partition)
      val tensor = Tensor[Float](10).fill(partitionID.toFloat + 1.0f)
      sync.init(s"testPara", 10, weights = null, grads = tensor)
      var res : Iterator[_] = null
      sync.put(s"testPara")
      res = Iterator.single(sync.get(s"testPara"))
      sync.clear
      res
    }).collect
    res.length should be  (4)
    res(0).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](10).fill(2.5f))
    res(1).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](10).fill(2.5f))
    res(2).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](10).fill(2.5f))
    res(3).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](10).fill(2.5f))
  }

  "DistributedSynchronizer with parameter size less than partition" should "work properly" in {
    val cores1 = Runtime.getRuntime().availableProcessors
    val partition = 4
    val cores = 4
    val res = sc.parallelize((0 until partition), partition).mapPartitions(p => {
      Engine.setNodeAndCore(partition, cores)
      val partitionID = TaskContext.getPartitionId
      val sync = new BlockManagerParameterSynchronizer[Float](partitionID, partition)
      val tensor = Tensor[Float](2).fill(partitionID.toFloat + 1.0f)
      sync.init(s"testPara", 2, weights = null, grads = tensor)
      var res : Iterator[_] = null
      sync.put(s"testPara")
      res = Iterator.single(sync.get(s"testPara"))
      sync.clear
      res
    }).collect
    res.length should be  (4)
    res(0).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](2).fill(2.5f))
    res(1).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](2).fill(2.5f))
    res(2).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](2).fill(2.5f))
    res(3).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](2).fill(2.5f))
  }

  "DistributedSynchronizer with parameter offset > 1" should "work properly" in {
    val partition = 4
    val cores = 4
    val res = sc.parallelize((0 until partition), partition).mapPartitions(p => {
      Engine.setNodeAndCore(partition, cores)
      val partitionID = TaskContext.getPartitionId
      val sync = new BlockManagerParameterSynchronizer[Float](partitionID, partition)
      val tensor = Tensor[Float](20)
      val parameter = tensor.narrow(1, 10, 10).fill(partitionID.toFloat + 1.0f)
      sync.init(s"testPara", 10, weights = null, grads = parameter)
      var res : Iterator[_] = null
      sync.put(s"testPara")
      res = Iterator.single(sync.get(s"testPara"))
      sync.clear
      res
    }).collect
    res.length should be  (4)
    res(0).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](10).fill(2.5f))
    res(1).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](10).fill(2.5f))
    res(2).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](10).fill(2.5f))
    res(3).asInstanceOf[Tuple2[_, _]]._2 should be (Tensor[Float](10).fill(2.5f))
  }

  after {
    sc.stop
  }
}
