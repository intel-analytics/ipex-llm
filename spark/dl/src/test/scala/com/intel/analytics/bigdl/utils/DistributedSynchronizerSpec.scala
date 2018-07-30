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
import org.scalatest.{FlatSpec, Matchers}

class DistributedSynchronizerSpec  extends FlatSpec with Matchers {
  "DistributedSynchronizer" should "work properly" in {
    val conf = Engine.createSparkConf().setAppName("test synchronizer").setMaster("local[*]")
      .set("spark.rpc.message.maxSize", "200")
    val sc = new SparkContext(conf)
    Engine.init
    val partition = 4
    val cores = 4
    val res = sc.parallelize((0 until partition), partition).mapPartitions(p => {
      Engine.setNodeAndCore(partition, cores)
      val partitionID = TaskContext.getPartitionId
      val sync = new BlockManagerParameterSynchronizer[Float](partitionID, partition)
      val tensor = Tensor[Float](10).fill(partitionID.toFloat + 1.0f)
      sync.init(s"testPara", 10)
      var res : Iterator[_] = null
      sync.put(s"testPara", tensor)
      // sync.get("testPara")
      res = Iterator.single(sync.get(s"testPara"))
      sync.clear
      res
    }).collect
    res.length should be  (4)
    res(0) should be (Tensor[Float](3).fill(2.5f))
    res(1) should be (Tensor[Float](3).fill(2.5f))
    res(2) should be (Tensor[Float](2).fill(2.5f))
    res(3) should be (Tensor[Float](2).fill(2.5f))
    sc.stop
  }
}
