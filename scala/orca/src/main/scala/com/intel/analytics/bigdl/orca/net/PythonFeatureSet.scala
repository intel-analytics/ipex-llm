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

package com.intel.analytics.bigdl.orca.net

import java.util

import java.nio.file.Paths
import java.util
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dllib.feature.dataset.{AbstractDataSet, DistributedDataSet, MiniBatch, Transformer}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator
import com.intel.analytics.bigdl.orca.utils.PythonInterpreter
import com.intel.analytics.bigdl.dllib.feature.common.{ArrayLike, ArrayLikeWrapper}
import com.intel.analytics.bigdl.dllib.feature.{FeatureSet, _}
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.slf4j.{Logger, LoggerFactory}
import jep._

import scala.reflect.ClassTag
import scala.collection.JavaConverters._

import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.reflect.ClassTag

object PythonFeatureSet{

  private[bigdl] def python[T: ClassTag](
      dataset: Array[Byte],
      getLoader: (Int, Int, String) => String,
      getIterator: (String, String, Boolean) => String,
      getNext: (String) => String,
      inputName: String,
      targetName: String,
      totalSize: Int,
      imports: String = "",
      loaderName: String = s"loader${Integer.toHexString(java.util.UUID.randomUUID().hashCode())}"
    ): PythonFeatureSet[T] = {
    new PythonFeatureSet[T](dataset, getLoader, getIterator, getNext,
      inputName, targetName, totalSize, imports)
  }

  // One partition one loader
  protected def getLocalLoader(loaderName: String): String = {
    s"${loaderName}_${TaskContext.getPartitionId()}"
  }

  protected def getLocalIter(loaderName: String, train: Boolean): String = {
    s"${loaderName}_iter_${train}"
  }

  protected def loadPythonSet(
                               loaderName: String,
                               getLoader: (Int, Int, String) => String,
                               dataset: Array[Byte],
                               imports: String,
                               interpRdd: RDD[Int]): Unit = {
    val bcDataSet = interpRdd.sparkContext.broadcast(dataset)
    val nodeNumber = Engine.nodeNumber()
    val preimports = s"""
                        |from pyspark.serializers import CloudPickleSerializer
                        |import numpy as np
                        |""".stripMargin + imports
    interpRdd.mapPartitions{iter =>
      val partId = TaskContext.getPartitionId()
      require(partId < nodeNumber, s"partId($partId) should be" +
        s" smaller than nodeNumber(${nodeNumber})")
      PythonInterpreter.exec(preimports)
      PythonInterpreter.set("pyjarray", bcDataSet.value)

      val localLoaderName = getLocalLoader(loaderName)

      val load = getLoader(nodeNumber, partId, localLoaderName)
      PythonInterpreter.exec(load)
      Iterator.single(1)
    }.count()
  }

  protected lazy val cachedRdd: RDD[Int] = createCachedRdd()
  protected def createCachedRdd(): RDD[Int] = {
    val sc = SparkContext.getOrCreate()
    val nodeNumber = Engine.nodeNumber()
    // TODO: make sure 1 executor 1 partition
    val originRdd = sc.parallelize(
      Array.tabulate(nodeNumber)(_ => "dummy123123"), nodeNumber * 10)
      .mapPartitions(_ => (0 until 20000000).toIterator)
      .coalesce(nodeNumber)
      .setName("PartitionRDD")
      .persist(StorageLevel.DISK_ONLY)
    originRdd.count()
    originRdd
  }

  private[bigdl] def toArrayTensor(
                                  data: AnyRef): Array[Tensor[Float]] = {
    data match {
      case ndArray: NDArray[_] =>
        Array(ndArrayToTensor(ndArray))
      case ndArrays: util.ArrayList[_] =>
        if (ndArrays.size() > 0) {
          ndArrays.get(0) match {
            case _: NDArray[_] =>
              ndArrays.asInstanceOf[util.ArrayList[NDArray[_]]].asScala.toArray.map { input =>
                ndArrayToTensor(input)
              }
            // TODO: support ArrayList[String]
          }
        } else {
          Array()
        }
      case _ =>
        throw new IllegalArgumentException(s"supported type ${data.getClass()}")
    }
  }

  private[bigdl] def ndArrayToTensor(ndArray: NDArray[_]): Tensor[Float] = {
    val array = ndArray.asInstanceOf[NDArray[Array[_]]]
    val data = array.getData()
    if (data.length > 0) {
      data(0) match {
        case _: Float =>
          Tensor[Float](data.asInstanceOf[Array[Float]], array.getDimensions)
        case _ =>
          Tensor[Float](data.map(_.toString.toFloat), array.getDimensions)
      }
    } else {
      Tensor[Float]()
    }
  }
}

class PythonFeatureSet[T: ClassTag](
                                     dataset: Array[Byte],
                                     getLoader: (Int, Int, String) => String,
                                     getIterator: (String, String, Boolean) => String,
                                     getNext: (String) => String,
                                     inputName: String,
                                     targetName: String = "",
                                     totalSize: Int,
                                     imports: String = "",
                                     loaderName: String = s"loader${Integer.toHexString(
                                       java.util.UUID.randomUUID().hashCode())}"
                                   ) extends DistributedFeatureSet[T] {
  import PythonFeatureSet._

  loadPythonSet(loaderName, getLoader, dataset, imports, cachedRdd)

  override def originRDD(): RDD[_] = {
    cachedRdd
  }

  override def data(train: Boolean): RDD[T] = {
    val loaderName = this.loaderName
    val inputName = this.inputName
    val targetName = this.targetName
    val getNext = this.getNext
    val getIterator = this.getIterator
    if (train) {
      cachedRdd.mapPartitions{dataIter =>
        val localLoaderName = getLocalLoader(loaderName)
        val localIterName = getLocalIter(localLoaderName, train)
        val getIteratorCode = getIterator(localIterName, localLoaderName, train)

        val nextCode = getNext(localIterName)
        new Iterator[T] {
          override def hasNext: Boolean = {
            true
          }

          override def next(): T = {
            try {
              PythonInterpreter.exec(nextCode)
            } catch {
              case e: Exception =>
                if (e.getMessage().contains("End of sequence") ||
                  e.getMessage().contains("StopIteration") ||
                  e.getMessage().contains("is not defined")) {
                  PythonInterpreter.exec(getIteratorCode)
                  PythonInterpreter.exec(nextCode)
                  FeatureSet.logger.debug("The data has been iterated. Start the next epoch...")
                } else {
                  throw e
                }
            }
            val inputs = toArrayTensor(PythonInterpreter.getValue[AnyRef](inputName))
            val miniBatch = if (targetName != "") {
              val targets = toArrayTensor(PythonInterpreter.getValue(targetName))
              MiniBatch[Float](inputs, targets)
            } else {
              MiniBatch[Float](inputs)
            }
            miniBatch.asInstanceOf[T]
          }
        }
      }
    } else {
      cachedRdd.mapPartitions{ dataIter =>
        val localLoaderName = getLocalLoader(loaderName)
        val localIterName = getLocalIter(localLoaderName, train)
        PythonInterpreter.exec(getIterator(localIterName, localLoaderName, train))
        new Iterator[T] {
          val nextCode = getNext(localIterName)
          var alreadyNext = false

          override def hasNext: Boolean = {
            if (!alreadyNext) {
              try {
                PythonInterpreter.exec(nextCode)
              } catch {
                case e: Exception =>
                  if (e.getMessage().contains("End of sequence") ||
                    e.getMessage().contains("StopIteration")) {
                    return false
                  } else {
                    throw e
                  }
              }
              alreadyNext = true
            }
            true
          }

          override def next(): T = {
            if (!alreadyNext) {
              PythonInterpreter.exec(nextCode)
            }
            val inputs = toArrayTensor(PythonInterpreter.getValue(inputName))
            val miniBatch = if (targetName != "") {
              val targets = toArrayTensor(PythonInterpreter.getValue(targetName))
              MiniBatch[Float](inputs, targets)
            } else {
              MiniBatch[Float](inputs)
            }
            alreadyNext = false
            miniBatch.asInstanceOf[T]
          }
        }

      }
    }

  }

  override def shuffle(): Unit = {

  }

  override def size(): Long = {
    data(false).count()
  }

  override def toDistributed(): DistributedDataSet[T] = {
    new DistributedDataSetWrapper[T](this)
  }
}
