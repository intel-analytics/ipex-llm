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

package com.intel.analytics.bigdl.orca.net.python

import java.util.concurrent.{CopyOnWriteArrayList, TimeUnit}
import java.util.{ArrayList, List => JList}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.utils.python.api.JTensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.common.PythonZoo
import com.intel.analytics.bigdl.dllib.utils.python.api.EvaluatedResult
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.net.NetUtils
import com.intel.analytics.bigdl.dllib.feature.dataset.MiniBatch
import com.intel.analytics.bigdl.dllib.optim.{LocalPredictor, ValidationMethod, _}
import com.intel.analytics.bigdl.orca.net._
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.config.Configurator
import org.apache.spark.api.java.JavaRDD

import scala.collection.JavaConverters._
import scala.reflect.ClassTag


object PythonZooNet {

  def ofFloat(): PythonZooNet[Float] = new PythonZooNet[Float]()

  def ofDouble(): PythonZooNet[Double] = new PythonZooNet[Double]()

}


class PythonZooNet[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  def tfnetEvaluate(model: AbstractModule[Activity, Activity, Float],
                    valRDD: JavaRDD[MiniBatch[Float]],
                    valMethods: JList[ValidationMethod[Float]])
  : JList[EvaluatedResult] = {
    val resultArray = TFNet.testMiniBatch(model, valRDD.rdd,
      valMethods.asScala.toArray)
    val testResultArray = resultArray.map { result =>
      EvaluatedResult(result._1.result()._1, result._1.result()._2,
        result._2.toString())
    }
    testResultArray.toList.asJava
  }

  def createTFNet(
                   path: String,
                   inputNames: JList[String],
                   outputNames: JList[String]): TFNet = {
    TFNet(path, inputNames.asScala.toArray, outputNames.asScala.toArray)
  }

  def createTFNet(
                   path: String,
                   inputNames: JList[String],
                   outputNames: JList[String], config: Array[Byte]): TFNet = {
    TFNet(path, inputNames.asScala.toArray, outputNames.asScala.toArray, config)
  }

  def createTFNet(path: String): TFNet = {
    TFNet(path)
  }

  def createTFNetFromSavedModel(path: String,
                                tag: String,
                                inputNames: JList[String],
                                outputNames: JList[String],
                                config: Array[Byte],
                                initOp: String): Module[Float] = {
    val sessionConfig = Option(config).getOrElse(TFNet.defaultSessionConfig.toByteArray())
    TFNetForInference.fromSavedModel(path, Option(tag), None,
      Option(inputNames.asScala.toArray), Option(outputNames.asScala.toArray),
      sessionConfig, Option(initOp))
  }

  def createTFNetFromSavedModel(path: String,
                                tag: String,
                                signature: String,
                                config: Array[Byte]): Module[Float] = {
    if (config == null) {
      TFNet.fromSavedModel(path, tag, signature)
    } else {
      TFNetForInference.fromSavedModel(path, Option(tag), Option(signature),
        None, None, config)
    }
  }

  def createTFNet(path: String, config: Array[Byte]): TFNet = {
    TFNet(path, config)
  }

  var processGpToBeKill: String = ""
  registerKiller()

  private def killPgid(pgid: String, killCommand: String): Boolean = {
    println("JVM is stopping process group: " +  pgid)
    val process = Runtime.getRuntime().exec(killCommand + pgid)
    process.waitFor(2, TimeUnit.SECONDS)
    process.exitValue() == 0
  }

  private def registerKiller(): Unit = {
    Configurator.setLevel("py4j.reflection.ReflectionEngine", Level.ERROR)
    Configurator.setLevel("py4j.GatewayConnection", Level.ERROR)
    Runtime.getRuntime().addShutdownHook(new Thread {
      override def run(): Unit = {
        if (processGpToBeKill == "") return
        // Give it a chance to be gracefully killed
        val success = killPgid(processGpToBeKill, "kill -- -")
        if (!success) {
          killPgid(processGpToBeKill, "kill -9 -")
        }
      }
    })
  }

  def jvmGuardRegisterPgid(gpid: Int): Unit = {
    this.processGpToBeKill = gpid.toString
  }

  def getModuleExtraParameters(model: AbstractModule[_, _, T]): Array[JTensor] = {
    model.getExtraParameter().map(toJTensor)
  }

  def createTorchModel(model: Array[Byte], weights: JTensor): TorchModel = {
    TorchModel(model, weights.storage)
  }

  def getTorchModelBytes(torchModel: TorchModel): Array[Byte] = {
    torchModel.modelHolder.torchBytes
  }

  def createTorchLoss(criterion: Array[Byte]): TorchLoss = {
    TorchLoss(criterion)
  }

  def createTorchOptim(optim: Array[Byte], decayType: String): TorchOptim[T] = {
    TorchOptim(optim, decayType)
  }

  def createFeatureSetFromTfDataset(
       dataset: Array[Byte],
       totalSize: Int): PythonFeatureSet[MiniBatch[Float]] = {
    val nodeNumber = Engine.nodeNumber()
    // set a random seed to make sure shuffle is the same in each executor
    val imports =
      s"""
        |import tensorflow as tf
        |from bigdl.dllib.utils.nest import flatten
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
    PythonFeatureSet.python[MiniBatch[Float]](dataset,
      getLoader, getIterator, getNext,
      "data", "", totalSize, imports)
  }

  def createFeatureSetFromPyTorch(
       dataloader: Array[Byte],
       creator: Boolean,
       features: String,
       labels: String): PythonFeatureSet[MiniBatch[Float]] = {
    val trainPostfix = "_train"
    val evalPostfix = "_eval"
    val loaderName: String =
      s"loader${Integer.toHexString(java.util.UUID.randomUUID().hashCode())}"
    val imports = s"""
                     |from bigdl.dllib.utils.nest import ptensor_to_numpy
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
      val brace = if (creator) "()" else ""
      val load = s"""
                    |by${partId} = bytes(b % 256 for b in pyjarray)
                    |func${partId} = CloudPickleSerializer.loads(CloudPickleSerializer,
                      by${partId})
                    |${localLoaderName} = func${partId}${brace}
                    |""".stripMargin
      load +
        s"""
           |from torch.utils.data.distributed import DistributedSampler
           |from torch.utils.data.sampler import RandomSampler
           |from bigdl.orca.torch.utils import DistributedSequentialSampler
           |from torch.utils.data import DataLoader
           |import math
           |
           |${localLoaderName}_rand_sampler=DistributedSampler(${localLoaderName}.dataset,
           |                                              ${nodeNumber}, ${partId}, True)
           |${localLoaderName}_seq_sampler=DistributedSequentialSampler(
           |  ${localLoaderName}.dataset,
           |                                              ${nodeNumber}, ${partId})
           |
           |${loaderName}_bs_node = int(math.ceil(${localLoaderName}.batch_size / ${nodeNumber}))
           |
           |data_loader_args = {
           |                "dataset": ${localLoaderName}.dataset,
           |                "batch_size": ${loaderName}_bs_node,
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
    val inputsName = if (features == null || features == "") {
      s"torch.Tensor(${loaderName}_bs_node, 1)"
    } else {
      features
    }
    val targetsName = if (labels == null || labels == "") {
      s"torch.Tensor(${loaderName}_bs_node, 1)"
    } else {
      labels
    }
    PythonFeatureSet.python[MiniBatch[Float]](dataloader, getLoader, getIterator, getNext,
        s"ptensor_to_numpy(${inputsName})",
        s"ptensor_to_numpy(${targetsName})", -1, imports)
  }
}
