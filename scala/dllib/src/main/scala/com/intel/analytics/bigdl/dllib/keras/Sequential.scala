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

package com.intel.analytics.bigdl.dllib.keras

import java.io.{File, FilenameFilter}
import java.text.SimpleDateFormat
import java.util.Calendar

import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.dllib.feature.dataset.{MiniBatch, _}
import com.intel.analytics.bigdl.dllib.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.dllib.feature.dataset.DataSet
import com.intel.analytics.bigdl.dllib.optim
import com.intel.analytics.bigdl.dllib._
import com.intel.analytics.bigdl.dllib.nn.Graph._
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.keras.{KerasLayer, KerasLayerSerializable}
import com.intel.analytics.bigdl.dllib.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.dllib.nn.{Container, Graph, Module, StaticGraph, Sequential => TSequential}
import com.intel.analytics.bigdl.dllib.optim.DistriOptimizer.{Cache, CacheV1}
import com.intel.analytics.bigdl.dllib.optim.DistriOptimizerV2.{Cache => CacheV2}
import com.intel.analytics.bigdl.dllib.optim._
import com.intel.analytics.bigdl.dllib.optim.parameters.AllReduceParameter
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils._
import com.intel.analytics.bigdl.dllib.utils.serializer.{DeserializeContext, ModuleData, ModuleSerializer, SerializeContext}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.optim.ZooTrigger
import com.intel.analytics.bigdl.dllib.feature.{DiskFeatureSet, DistributedFeatureSet, FeatureSet}
import com.intel.analytics.bigdl.dllib.feature.image.ImageSet
import com.intel.analytics.bigdl.dllib.feature.dataset
import com.intel.analytics.bigdl.dllib.feature.text._
import com.intel.analytics.bigdl.dllib.keras.{Net, Predictable}
import com.intel.analytics.bigdl.dllib.keras.autograd.{Lambda, Variable}
import com.intel.analytics.bigdl.dllib.keras.autograd._
import com.intel.analytics.bigdl.dllib.keras.layers.Input
import com.intel.analytics.bigdl.dllib.keras.layers.utils._
import com.intel.analytics.bigdl.dllib.keras.models._
import com.intel.analytics.bigdl.dllib.net.NetUtils
// import com.intel.analytics.bigdl.dllib.Net.TorchModel
import com.intel.analytics.bigdl.dllib.estimator.{AbstractEstimator, ConstantClipping, GradientClipping, L2NormClipping}
// import com.intel.analytics.zoo.tfpark.{TFTrainingHelper, TFTrainingHelperV2}
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.commons.lang3.SerializationUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.log4j.Logger
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.rdd.{RDD, ZippedPartitionsWithLocalityRDD}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.language.implicitConversions

class Sequential[T: ClassTag] private ()
  (implicit ev: TensorNumeric[T]) extends KerasNet[T] {

  private[bigdl] var frozen: Boolean = false

  this.labor = doBuild(null)

  private def buildModule(module: AbstractModule[_ <: Activity, _ <: Activity, T]): Unit = {
    val absModuleRef =
      new AbstractModuleRef(module.asInstanceOf[AbstractModule[Activity, Activity, T]])
    val kerasLayerRef = KerasLayerRef(this)

    if (!this.isBuilt()) {
      if (module.getInputShape() == null) {
        throw new RuntimeException("The first layer should explicitly declare inputshape")
      } else {

        val outputShape = absModuleRef.build(module.getInputShape())
        // The inputShape of Sequential should only be init here.
        kerasLayerRef.setInputShape(module.getInputShape())
        kerasLayerRef.setOutShape(outputShape)
      }
    } else {
      val outputShape = absModuleRef.build(this.getOutputShape())
      kerasLayerRef.setOutShape(outputShape)
    }
  }

  private def getLambdaLayer(lambda: Lambda[T]):
  AbstractModule[_ <: Activity, _ <: Activity, T] = {
    val inputShape = if (!this.isBuilt()) {
      if (lambda.getInputShape() == null) {
        throw new RuntimeException("The first layer should explicitly declare inputshape")
      }
      lambda.getInputShape()
    } else {
      this.getOutputShape()
    }
    return lambda.create(
      KerasUtils.removeBatch(inputShape))
  }

  def add(lambda: Lambda[T]): Sequential[T] = {
    add(getLambdaLayer(lambda))
  }

  /**
   * Add a sub-module to the sequential container.
   *
   * @param module The module to be added.
   * @return This sequential container.
   */
  def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    if (frozen) {
      throw new RuntimeException(
        "This Sequential has been frozen, as it has been added into other container")
    }

    if (module.isInstanceOf[Sequential[T]]) {
      module.asInstanceOf[Sequential[T]].frozen = true
    }
    val mModule = module
    val kerasLayerRef = KerasLayerRef(this)
    kerasLayerRef.validateInput[T](Seq(mModule))

    buildModule(mModule)

    labor.asInstanceOf[TSequential[T]].modules +=
      mModule.asInstanceOf[AbstractModule[Activity, Activity, T]]
    kerasLayerRef.checkDuplicate()
    this
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    if (labor.asInstanceOf[TSequential[T]].modules.isEmpty) {
      inputShape
    } else {
      labor.asInstanceOf[TSequential[T]].modules.last.getOutputShape()
    }
  }

  override def doBuild(inputShape: Shape): TSequential[T] = TSequential[T]()

  override def build(calcInputShape: Shape): Shape = {
    val kerasLayerRef = KerasLayerRef(this)
    kerasLayerRef.checkWithCurrentInputShape(calcInputShape)
    getOutputShape()
  }

  override def toModel(): Model[T] = {
    val input = Input[T](KerasUtils.removeBatch(this.getInputShape()))

    // the is reason we do not use .inputs here is
    // layers in modules cannot be rebuilt
    val output = this.modules(0)
      .asInstanceOf[TSequential[T]]
      .modules.foldLeft(input) { (i1, i2) =>
      val out = Node(i2)
      i1.add(out, Edge())
      out
    }
    Model(input, output)
  }

  override def summary(
                        lineLength: Int = 120,
                        positions: Array[Double] = Array(.33, .55, .67, 1)): Unit = {
    val graph = this.toModel()
    graph.summary(lineLength, positions)
  }

  override private[bigdl] def getKerasWeights(): Array[Tensor[Float]] = {
    val weights = new ArrayBuffer[Tensor[Float]]()
    modules(0).asInstanceOf[TSequential[T]].modules.foreach(m => {
      val params = m.asInstanceOf[Net].getKerasWeights()
      if (params != null) {
        params.foreach{p =>
          weights += p
        }
      }
    })
    weights.toArray
  }
}

object Sequential extends KerasLayerSerializable {
  ModuleSerializer.registerModule(
    "com.intel.analytics.bigdl.dllib.keras.Sequential",
    Sequential)

  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : Sequential[T] = {
    new Sequential[T]()
  }
}
