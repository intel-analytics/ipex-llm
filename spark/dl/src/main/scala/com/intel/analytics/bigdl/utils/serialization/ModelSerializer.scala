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
package com.intel.analytics.bigdl.utils.serialization

import scala.collection.JavaConverters._

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{L1L2Regularizer, L1Regularizer, L2Regularizer, Regularizer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import serialization.Model
import serialization.Model.BigDLModel.ModuleType
import serialization.Model._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object ModelSerializer {

  private val serializerMap = new mutable.HashMap[String, AbstractModelSerializer]()
  private val hdfsPrefix: String = "hdfs:"

  init

  case object AbsSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      createBigDLModule(model, Abs().asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val abs = module.module.asInstanceOf[Abs[T]]
      bigDLModelBuilder.setModuleType(ModuleType.ABS)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object AddSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val addParam = model.getAddParam
      val inputSize = addParam.getInputSize
      val add = Add[T](inputSize)
      createBigDLModule(model, add.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val add = module.module.asInstanceOf[Add[T]]
      val inputSize = add.inputSize
      val addParam = AddParam.newBuilder
      addParam.setInputSize(inputSize)
      bigDLModelBuilder.setModuleType(ModuleType.ADD)
      bigDLModelBuilder.setAddParam(addParam.build)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object LinearSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
      : BigDLModule[T] = {
      createBigDLModule(model, createLinear(model).
        asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
      (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val linear = module.module.asInstanceOf[Linear[T]]
      createSerializeLinear(bigDLModelBuilder, linear)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
    private def createLinear[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T]):
      Linear[T] = {
      val linearNarams = model.getLinearParam
      val inputSize = linearNarams.getInputSize
      val outputSize = linearNarams.getOutputSize
      var initMethod : InitializationMethod = null
      if (linearNarams.hasInitMethod) {
        initMethod = createInitMethod(linearNarams.getInitMethod)
      }
      val withBias = if (linearNarams.hasWithBias) linearNarams.getWithBias else true
      var wRegularizer : Regularizer[T] = null
      if (linearNarams.hasWRegularizer) {
        wRegularizer = createRegularizer(linearNarams.getWRegularizer)
      }
      var bRegularizer : Regularizer[T] = null
      if (linearNarams.hasBRegularizer) {
        bRegularizer = createRegularizer(linearNarams.getBRegularizer)
      }
      Linear[T](inputSize, outputSize, initMethod, withBias, wRegularizer, bRegularizer)
    }

    private def createSerializeLinear[T: ClassTag](
      modelBuilder : BigDLModel.Builder, linear : Linear[T])
      (implicit ev: TensorNumeric[T]): Unit = {
      val linearParam = LinearParam.newBuilder
      linearParam.setInputSize(linear.inputSize)
      linearParam.setOutputSize(linear.outputSize)
      linearParam.setWithBias(linear.withBias)
      if (linear.wRegularizer != null) {
        linearParam.setWRegularizer(createSerializeRegularizer(linear.wRegularizer))
      }
      if (linear.bRegularizer != null) {
        linearParam.setBRegularizer(createSerializeRegularizer(linear.bRegularizer))
      }
      linearParam.setInitMethod(createSerializeInitMethod(linear.initMethod))
      modelBuilder.setLinearParam(linearParam.build)
      modelBuilder.setModuleType(ModuleType.LINEAR)
    }
  }

  case object SequentialSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val subModules = model.getSubModulesList.asScala
      val sequantial = Sequential[T]()
      subModules.foreach(subModule => {
        val bigDLModule = load(subModule)
        sequantial.add(bigDLModule.module.
          asInstanceOf[AbstractModule[_ <: Activity, _ <: Activity, T]])
      })
      createBigDLModule(model, sequantial)
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
      (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      module.bottoms.foreach(_ => bigDLModelBuilder.addAllBottoms(_))
      module.tops.foreach(_ => bigDLModelBuilder.addTops(_))
      bigDLModelBuilder.setName(module.module.getName)
      val sequential = module.module.asInstanceOf[Sequential[T]]
      sequential.modules.foreach(subModule => {
        val subModel = serialize(BigDLModule(subModule, module.tops, module.bottoms))
        bigDLModelBuilder.addSubModules(subModel)
      })
      bigDLModelBuilder.setModuleType(ModuleType.SEQUENTIAL)
      bigDLModelBuilder.build
    }
  }

  case object GraphSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val subModules = model.getSubModulesList.asScala
      val modules = new ArrayBuffer[ModuleNode[T]]()
      // map all bottom modules to current module
      val bottomToModules = new mutable.HashMap[String, ModuleNode[T]]()
      subModules.foreach(subModule => {
        val bigDLModule = load(subModule)
        val moduleNode = bigDLModule.module.apply()
        val tops = bigDLModule.tops
        tops.foreach(top => {
          if (bottomToModules.contains(top)) {
            bottomToModules(top) -> moduleNode
          }
        })
        val bottoms = bigDLModule.bottoms
        bottoms.foreach(bottom => bottomToModules(bottom) = moduleNode)
        modules.append(moduleNode)
      })
      val inputs = modules.filter(_.prevNodes.size == 0).toArray
      val outputs = modules.filter(_.nextNodes.size == 0).toArray
      val graph = Graph[T](inputs, outputs)
      createBigDLModule(model, graph)
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      module.bottoms.foreach(_ => bigDLModelBuilder.addAllBottoms(_))
      module.tops.foreach(_ => bigDLModelBuilder.addTops(_))
      bigDLModelBuilder.setName(module.module.getName)
      val graph = module.module.asInstanceOf[Graph[T]]
      graph.getExecutions.foreach(execution => {
        val tops = execution.prevNodes.map(_.element.getName)
        val bottoms = execution.nextNodes.map(_.element.getName)
        val subModel = serialize(BigDLModule(execution.element
            .asInstanceOf[AbstractModule[Activity, Activity, T]],
          tops, bottoms))
        bigDLModelBuilder.addSubModules(subModel)
      })
      bigDLModelBuilder.setModuleType(ModuleType.GRAPH)
      bigDLModelBuilder.build
    }
  }

  def load[T: ClassTag](model: BigDLModel)
    (implicit ev: TensorNumeric[T]) : BigDLModule[T] = {
    serializerMap(model.getModuleType.toString).loadModule(model)
  }

  def serialize[T: ClassTag](bigDLModule : BigDLModule[T])
    (implicit ev: TensorNumeric[T])
    : BigDLModel = {
    val module = bigDLModule.module.asInstanceOf[AbstractModule[_, _, _]]
    val bigDLModel = module match {
      case abs : Abs[_] => AbsSerializer.serializeModule(bigDLModule)
      case add : Add[_] => AddSerializer.serializeModule(bigDLModule)
      case linear : Linear[_] => LinearSerializer.serializeModule(bigDLModule)
      case sequantial : Sequential[_] => SequentialSerializer.serializeModule(bigDLModule)
      case graph : Graph[_] => GraphSerializer.serializeModule(bigDLModule)
      case _ => throw new IllegalArgumentException(s"$module serialization is not supported")
    }
    bigDLModel
  }

  private  def init(): Unit = {
    serializerMap("ABS") = AbsSerializer
    serializerMap("ADD") = AddSerializer
    serializerMap("LINEAR") = LinearSerializer
    serializerMap("SEQUENTIAL") = SequentialSerializer
    serializerMap("GRAPH") = GraphSerializer
  }

}
