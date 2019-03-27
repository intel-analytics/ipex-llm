/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.feature.pmem

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.examples.inception.ImageNet2012
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import org.apache.spark.SparkContext
import org.scalatest.Ignore

import scala.collection.mutable.ArrayBuffer

@Ignore
class PersistentMemorySpec extends ZooSpecHelper {
  var sc: SparkContext = null

  override def doBefore(): Unit = {
    val conf = Engine.createSparkConf().setAppName("PersistentMemorySpec")
      .set("spark.task.maxFailures", "1").setMaster("local[4]")
    sc = NNContext.initNNContext(conf)
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "load native lib optanedc" should "be ok" in {
    val address = MemoryAllocator.getInstance(PMEM).allocate(1000L)
    MemoryAllocator.getInstance(PMEM).free(address)
  }

  "NativeFloatArray optane dc" should "be ok" in {
    val array = Array[Float](1.2f, 0.3f, 4.5f, 199999.6f)
    val floatArray = FloatArray(array.toIterator, array.size)
    var i = 0
    while( i < floatArray.recordNum) {
      assert(floatArray.get(i) == array(i))
      i += 1
    }
    floatArray.free()
  }

  "NativeBytesArray optanedc" should "be ok" in {
    val sizeOfItem = 100
    val sizeOfRecord = 5
    val nativeArray = new FixLenBytesArray(sizeOfItem, sizeOfRecord, PMEM)
    val targetArray = ArrayBuffer[Byte]()
    val rec = Array[Byte](193.toByte, 169.toByte, 0, 90, 4)
    (0 until 100).foreach {i =>
      nativeArray.set(i, rec)
    }

    var i = 0
    while( i < sizeOfItem) {
      assert(nativeArray.get(i) === rec)
      i += 1
    }
    nativeArray.free()
  }

  "NativevarBytesArray optane dc" should "be ok" in {
    val nativeArray = new VarLenBytesArray(3, 5 + 2 + 6, PMEM)
    val targetArray = ArrayBuffer[Byte]()
    val rec0 = Array[Byte](193.toByte, 169.toByte, 0, 90, 4)
    val rec1 = Array[Byte](90, 4)
    val rec2 = Array[Byte](193.toByte, 169.toByte, 0, 90, 4, 5)

    nativeArray.set(0, rec0)
    nativeArray.set(1, rec1)
    nativeArray.set(2, rec2)

    assert(nativeArray.get(0) === rec0)
    assert(nativeArray.get(1) === rec1)
    assert(nativeArray.get(2) === rec2)
    nativeArray.free()
  }

  "NativevarFloatsArray dram" should "be ok" in {
    val nativeArray = new VarLenFloatsArray(3, (5 + 2 + 6) * 4, PMEM)
    val targetArray = ArrayBuffer[Byte]()
    val rec0 = Array[Float](1.2f, 1.3f, 0, 0.1f, 0.2f)
    val rec1 = Array[Float](0.9f, 4.0f)
    val rec2 = Array[Float](193.5f, 169.4f, 0.0f, 90.1f, 4.3f, 5.6f)

    nativeArray.set(0, rec0)
    nativeArray.set(1, rec1)
    nativeArray.set(2, rec2)

    assert(nativeArray.get(0) === rec0)
    assert(nativeArray.get(1) === rec1)
    assert(nativeArray.get(2) === rec2)
    nativeArray.free()
  }

  "cached imageset optanedc" should "be ok" in {

    val dataPath = getClass.getClassLoader.getResource("pmem/mini_imagenet_seq").getPath

    val imageNet = ImageNet2012(path = dataPath,
      sc = sc,
      imageSize = 224,
      batchSize = 2,
      nodeNumber = 1,
      coresPerNode = 4,
      classNumber = 1000,
      memoryType = PMEM).asInstanceOf[DistributedDataSet[MiniBatch[Float]]]
    val data = imageNet.data(train = false)
    assert(data.count() == 3)
    data.collect()
  }

  "getting data in FeatureSet" should "be right" in {
    val samples = sc.range(1, 10).map(v =>
      Sample[Float](Tensor[Float].range(v, v + 5), v))
    val featureSet = FeatureSet.rdd(samples, memoryType = PMEM)
    featureSet.shuffle()
    val dataIter = featureSet.data(false)
    val data = dataIter.mapPartitions(v =>
      v.map(s => (s.feature(), s.label().valueAt(1)))
    ).collect()
    data.map(_._2.toInt).sorted should be (Array.range(1, 10))
    data.foreach(d =>
      d._1 should be (Tensor[Float].range(d._2, d._2 + 5))
    )
  }

  "getting ImageFeature from FeatureSet" should "be right" in {
    val bytes = Array.tabulate(10)(v => (Array.range(v, v + 40).map(_.toByte), v))
    val samples = sc.parallelize(bytes.map(v => ImageFeature(v._1, v._2))).setName("Example Sample")
    val featureSet = FeatureSet.rdd(samples, memoryType = PMEM)
    featureSet.shuffle()
    val dataIter = featureSet.data(false)
    val data = dataIter.mapPartitions(v =>
      v.map(s => (s.bytes(), s.getLabel[Int]))
    ).collect()
    data.map(_._2.toInt).sorted should be(Array.range(0, 10))
    data.foreach(d =>
      d._1 should be(Array.range(d._2, d._2 + 40).map(_.toByte))
    )
  }
}
