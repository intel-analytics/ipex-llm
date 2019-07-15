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
package com.intel.analytics.zoo.common

import java.util

import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDLKeras, Sample}
import com.intel.analytics.bigdl.tensor.{DenseType, SparseType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.Predictable
import org.apache.spark.api.java.JavaRDD
import java.util.{List => JList}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.LocalPredictor
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.feature.text.TextSet
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonZoo {

  def ofFloat(): PythonZoo[Float] = new PythonZoo[Float]()

  def ofDouble(): PythonZoo[Double] = new PythonZoo[Double]()

}


class PythonZoo[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDLKeras[T] {

  private val typeName = {
    val cls = implicitly[ClassTag[T]].runtimeClass
    cls.getSimpleName
  }

  override def toTensor(jTensor: JTensor): Tensor[T] = {
    if (jTensor == null) return null

    this.typeName match {
      case "float" =>
        if (null == jTensor.indices) {
          if (jTensor.shape == null || jTensor.shape.product == 0) {
            Tensor()
          } else {
            Tensor(jTensor.storage.map(x => ev.fromType(x)), jTensor.shape)
          }
        } else {
          Tensor.sparse(jTensor.indices, jTensor.storage.map(x => ev.fromType(x)), jTensor.shape)
        }
      case "double" =>
        if (null == jTensor.indices) {
          if (jTensor.shape == null || jTensor.shape.product == 0) {
            Tensor()
          } else {
            Tensor(jTensor.storage.map(x => ev.fromType(x.toDouble)), jTensor.shape)
          }
        } else {
          Tensor.sparse(jTensor.indices,
            jTensor.storage.map(x => ev.fromType(x.toDouble)), jTensor.shape)
        }
      case t: String =>
        throw new IllegalArgumentException(s"Not supported type: ${t}")
    }
  }

  override def toJTensor(tensor: Tensor[T]): JTensor = {
    // clone here in case the the size of storage larger then the size of tensor.
    require(tensor != null, "tensor cannot be null")
    tensor.getTensorType match {
      case SparseType =>
        // Note: as SparseTensor's indices is inaccessible here,
        // so we will transfer it to DenseTensor. Just for testing.
        if (tensor.nElement() == 0) {
          JTensor(Array(), Array(0), bigdlType = typeName)
        } else {
          val cloneTensor = Tensor.dense(tensor)
          val result = JTensor(cloneTensor.storage().array().map(i => ev.toType[Float](i)),
            cloneTensor.size(), bigdlType = typeName)
          result
        }
      case DenseType =>
        if (tensor.nElement() == 0) {
          if (tensor.dim() == 0) {
            JTensor(null, null, bigdlType = typeName)
          } else {
            JTensor(Array(), tensor.size(), bigdlType = typeName)
          }
        } else {
          val cloneTensor = tensor.clone()
          val result = JTensor(cloneTensor.storage().array().map(i => ev.toType[Float](i)),
            cloneTensor.size(), bigdlType = typeName)
          result
        }
      case _ =>
        throw new IllegalArgumentException(s"toJTensor: Unsupported tensor type" +
          s" ${tensor.getTensorType}")
    }
  }

  def activityToList(outputActivity: Activity): JList[Object] = {
    if (outputActivity.isInstanceOf[Tensor[T]]) {
      val list = new util.ArrayList[Object]()
      list.add(toJTensor(outputActivity.toTensor))
      list
    } else {
      table2JList(outputActivity.toTable)
    }
  }

  private def table2JList(t: Table): JList[Object] = {
    var i = 1
    val list = new util.ArrayList[Object]()
    while (i <= t.length()) {
      val item = t[Object](i)
      if (item.isInstanceOf[Tensor[T]]) {
        list.add(toJTensor(item.asInstanceOf[Tensor[T]]))
      } else if (item.isInstanceOf[Table]) {
        list.add(table2JList(item.asInstanceOf[Table]))
      } else {
        throw new IllegalArgumentException(s"Table contains unrecognizable objects $item")
      }
      i += 1
    }
    list
  }

  def zooPredict(
                  module: Predictable[T],
                  x: JavaRDD[Sample],
                  batchPerThread: Int): JavaRDD[JList[Object]] = {
    module.predict(x.rdd.map(toJSample), batchPerThread).map(activityToList).toJavaRDD()
  }

  def zooForward(model: AbstractModule[Activity, Activity, T],
                 input: JList[JTensor],
                 inputIsTable: Boolean): JList[Object] = {
    val inputActivity = jTensorsToActivity(input, inputIsTable)
    val outputActivity = model.forward(inputActivity)
    val result = activityToList(outputActivity)
    result
  }

  def zooPredict(
                  module: Module[T],
                  x: JList[JTensor],
                  batchPerThread: Int): JList[JList[Object]] = {
    val sampleArray = toSampleArray(x.asScala.toList.map{f => toTensor(f)})
    val localPredictor = LocalPredictor(module,
      batchPerCore = batchPerThread)
    val result = localPredictor.predict(sampleArray)
    val finalResult = result.map(activityToList).toList.asJava
    finalResult
  }

  def zooPredict(
                  module: Predictable[T],
                  x: ImageSet,
                  batchPerThread: Int): ImageSet = {
    module.predict(x, batchPerThread)
  }

  def zooPredict(
                  module: Predictable[T],
                  x: TextSet,
                  batchPerThread: Int): TextSet = {
    module.predict(x, batchPerThread)
  }

  def zooPredictClasses(
                         module: Predictable[T],
                         x: JavaRDD[Sample],
                         batchPerThread: Int,
                         zeroBasedLabel: Boolean = true): JavaRDD[Int] = {
    module.predictClasses(toJSample(x), batchPerThread, zeroBasedLabel).toJavaRDD()
  }


  def setCoreNumber(num: Int): Unit = {
    EngineRef.setCoreNumber(num)
  }


}
