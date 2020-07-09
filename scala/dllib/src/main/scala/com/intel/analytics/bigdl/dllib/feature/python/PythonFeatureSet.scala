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

package com.intel.analytics.zoo.feature.python

import java.util

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.{MiniBatch, Transformer, Sample => JSample}
import com.intel.analytics.bigdl.python.api.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.feature.pmem.MemoryType
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.spark.api.java.JavaRDD

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonFeatureSet {

  def ofFloat(): PythonFeatureSet[Float] = new PythonFeatureSet[Float]()

  def ofDouble(): PythonFeatureSet[Double] = new PythonFeatureSet[Double]()
}

class PythonFeatureSet[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  def createFeatureSetFromImageFrame(
        imageFrame: ImageFrame,
        memoryType: String,
        sequentialOrder: Boolean, shuffle: Boolean): FeatureSet[ImageFeature] = {
    require(imageFrame.isDistributed(), "Only support distributed ImageFrame")
    FeatureSet.rdd(imageFrame.toDistributed().rdd, MemoryType.fromString(memoryType),
      sequentialOrder = sequentialOrder, shuffle = shuffle)
  }

  def createFeatureSetFromRDD(
        data: JavaRDD[Any],
        memoryType: String,
        sequentialOrder: Boolean,
        shuffle: Boolean): FeatureSet[Any] = {
    FeatureSet.rdd(data, MemoryType.fromString(memoryType),
      sequentialOrder = sequentialOrder, shuffle = shuffle)
  }

  def createSampleFeatureSetFromRDD(data: JavaRDD[Sample],
                                    memoryType: String,
                                    sequentialOrder: Boolean,
                                    shuffle: Boolean)
  : FeatureSet[JSample[T]] = {
    FeatureSet.rdd(toJSample(data),
      MemoryType.fromString(memoryType),
      sequentialOrder = sequentialOrder,
      shuffle = shuffle)
  }

  def transformFeatureSet(featureSet: FeatureSet[Any],
                       transformer: Transformer[Any, Any]): FeatureSet[Any] = {
    featureSet -> transformer
  }

  def featureSetToDataSet(featureSet: FeatureSet[Any]): DataSet[Any] = {
    featureSet.toDataSet()
  }

  def createFeatureSetFromTfDataset(
        dataset: Array[Byte],
        totalSize: Int): FeatureSet[MiniBatch[Float]] = {
    val nodeNumber = EngineRef.getNodeNumber()
    // set a random seed to make sure shuffle is the same in each executor
    val imports =
      s"""
        |import tensorflow as tf
        |from zoo.util.nest import flatten
        |sess = tf.Session()
        |""".stripMargin
    def getIterator(iterName: String, loaderName: String, train: Boolean): String = {
      s"""
         |${iterName} = ${loaderName}.make_one_shot_iterator()
         |""".stripMargin
    }
    def getLoader(nodeNumber: Int, partId: Int, localLoaderName: String): String = {
      s"""
         |by${partId} = bytes(b % 256 for b in pyjarray)
         |func${partId} = CloudPickleSerializer.loads(CloudPickleSerializer, by${partId})
         |${localLoaderName} = func${partId}().shard(${nodeNumber}, ${partId})
         |""".stripMargin
    }
    def getNext(iterName: String): String = {
      s"""
        |data = sess.run(${iterName}.get_next())
        |data = flatten(data)
        |""".stripMargin
    }
    FeatureSet.python[MiniBatch[Float]](dataset,
      getLoader, getIterator, getNext,
      "data", "", totalSize, imports)
  }

  def createFeatureSetFromPyTorch(
        dataloader: Array[Byte]): FeatureSet[MiniBatch[Float]] = {
    val trainPostfix = "_train"
    val evalPostfix = "_eval"
    val imports = s"""
                     |from zoo.util.nest import ptensor_to_numpy
                     |import torch
                     |from torch.utils.data import DataLoader
                     |
                     |""".stripMargin

    def getIterator(iterName: String, loaderName: String, train: Boolean): String = {
      if (train) {
        s"""
           |if '${loaderName}_epoch' not in dir():
           |  ${loaderName}_epoch = 0
           |else:
           |  ${loaderName}_epoch += 1
           |${loaderName}_rand_sampler.set_epoch(${loaderName}_epoch)
           |${iterName} = enumerate(${loaderName}${trainPostfix})
           |""".stripMargin
      } else {
        s"${iterName} = enumerate(${loaderName}${evalPostfix})"
      }
    }

    def getNext(iterName: String): String = {
      // _index and _data will used in TorchModel and TorchLoss
      s"""
         |_index, _data = next(${iterName})
         |""".stripMargin
    }

    def getLoader(nodeNumber: Int, partId: Int, localLoaderName: String): String = {
      val load = s"""
                    |by${partId} = bytes(b % 256 for b in pyjarray)
                    |func${partId} = CloudPickleSerializer.loads(CloudPickleSerializer, by${partId})
                    |${localLoaderName} = func${partId}
                    |""".stripMargin
      load +
        s"""
           |from torch.utils.data.distributed import DistributedSampler
           |from torch.utils.data.sampler import RandomSampler
           |from zoo.pipeline.api.torch.utils import DistributedSequentialSampler
           |from torch.utils.data import DataLoader
           |import math
           |
           |${localLoaderName}_rand_sampler=DistributedSampler(${localLoaderName}.dataset,
           |                                              ${nodeNumber}, ${partId}, True)
           |${localLoaderName}_seq_sampler=DistributedSequentialSampler(${localLoaderName}.dataset,
           |                                              ${nodeNumber}, ${partId})
           |
           |bs_node = int(math.ceil(${localLoaderName}.batch_size / ${nodeNumber}))
           |
           |data_loader_args = {
           |                "dataset": ${localLoaderName}.dataset,
           |                "batch_size": bs_node,
           |                "shuffle": False,
           |                "num_workers": 0,
           |                "collate_fn": ${localLoaderName}.collate_fn,
           |                "drop_last": ${localLoaderName}.drop_last,
           |                "timeout": ${localLoaderName}.timeout,
           |                "worker_init_fn": ${localLoaderName}.worker_init_fn,
           |                "sampler": ${localLoaderName}_rand_sampler
           |            }
           |${localLoaderName}${trainPostfix} = DataLoader(**data_loader_args)
           |data_loader_args["sampler"] = ${localLoaderName}_seq_sampler
           |${localLoaderName}${evalPostfix} = DataLoader(**data_loader_args)
           |""".stripMargin
    }

    FeatureSet.python[MiniBatch[Float]](dataloader, getLoader, getIterator, getNext,
      "ptensor_to_numpy(_data[0])", "ptensor_to_numpy(_data[1])", -1, imports)
  }

}
