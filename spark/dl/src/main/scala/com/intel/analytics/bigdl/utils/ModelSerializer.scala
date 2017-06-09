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

import java.io._

import com.google.protobuf.CodedInputStream
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{L1L2Regularizer, L1Regularizer, L2Regularizer, Regularizer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.IOUtils
import serialization.Model
import serialization.Model.BigDLModel.ModuleType
import serialization.Model._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

sealed abstract class ModelSerializer {

  def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T]) : BigDLModule[T]

  def serializeModule[T: ClassTag](module : BigDLModule[T])
                                  (implicit ev: TensorNumeric[T]) : BigDLModel

  /**
   * copy serialized data (weight and bias if exist) to BigDL module
   * @param model serialized module
   * @param module  bigDL Module with relationships
   */
  def copy2BigDL[T: ClassTag](model : BigDLModel, module : BigDLModule[T])
                             (implicit ev: TensorNumeric[T]): Unit = {
    val paramTable : Table = module.module.getParametersTable
    if (paramTable != null && paramTable.contains(model.getName)) {
      val modulePramTable : Table = paramTable(module.module.getName)
      val weight : Tensor[T] = if (modulePramTable.contains("weight")) {
        modulePramTable("weight") }
       else null
      val bias : Tensor[T] = if (modulePramTable.contains("bias")) {
        modulePramTable("bias") }
        else null
      if (weight != null) copy2BigDLTensor(weight, model.getWeight)
      if (bias != null) copy2BigDLTensor(bias, model.getBias)
    }
  }

  private def copy2BigDLTensor[T: ClassTag](tensor : Tensor[T], serializedTensor : BigDLTensor)
                                 (implicit ev: TensorNumeric[T]) : Unit = {
    val serializedData = serializedTensor.getDataList
    require(tensor.nElement() == serializedData.size(), "data size is not equal")
    var i = 0
    val tensorData = tensor.storage().array()
    var offset = tensor.storageOffset() - 1
    while (i < serializedData.size()) {
      tensorData(offset) = ev.fromType[Double](serializedData.get(i))
      offset += 1
      i += 1
    }
  }

  /**
    * copy BigDL module data (weight and bias if exist) to BigDL Model to be persisted
    * @param modelBuilder serialized module builder
    * @param module  bigDL Module with relationships
    */
  def copyFromBigDL[T: ClassTag](module : BigDLModule[T], modelBuilder : BigDLModel.Builder)
                      (implicit ev : TensorNumeric[T]) : Unit = {
    val paramTable : Table = module.module.getParametersTable
    if (paramTable != null && paramTable.contains(module.module.getName)) {
      val modulePramTable : Table = paramTable(module.module.getName)
      val weight : Tensor[T] = if (modulePramTable.contains("weight")) {
        modulePramTable("weight") }
      else null
      val bias : Tensor[T] = if (modulePramTable.contains("bias")) {
        modulePramTable("bias") }
      else null
      if (weight != null) {
        val weightTensorBuilder = BigDLTensor.newBuilder
        copyFromBigDLTensor(weight, weightTensorBuilder)
        modelBuilder.setWeight(weightTensorBuilder.build)
      }
      if (bias != null) {
        val biasTensorBuilder = BigDLTensor.newBuilder
        copyFromBigDLTensor(bias, biasTensorBuilder)
        modelBuilder.setBias(biasTensorBuilder.build)
      }
    }
  }

  private def copyFromBigDLTensor[T: ClassTag](tensor : Tensor[T],
    serializedTensor : BigDLTensor.Builder)(implicit ev: TensorNumeric[T]) : Unit = {
    var i = 0
    val tensorData = tensor.storage().array()
    var offset = tensor.storageOffset() - 1
    while (i < tensorData.length) {
      serializedTensor.addData(ev.toType[Double](tensorData(i)))
      i += 1
    }
    tensor.size().foreach(_ => serializedTensor.addSize(_))
  }

  protected def createBigDLModule[T: ClassTag](model : BigDLModel,
    module : AbstractModule[Activity, Activity, T])(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
    val tops = model.getTopsList.asScala
    val bottoms = model.getBottomsList.asScala
    val bigDLModule = BigDLModule(module, tops, bottoms)
    module.setName(model.getName)
    copy2BigDL(model, bigDLModule)
    bigDLModule
  }

  protected def createSerializeBigDLModule[T: ClassTag](
    modelBuilder : BigDLModel.Builder, module : BigDLModule[T])(implicit ev: TensorNumeric[T])
  : BigDLModel = {
    module.bottoms.foreach(bottom => modelBuilder.addBottoms(bottom))
    module.tops.foreach(top => modelBuilder.addTops(top))
    modelBuilder.setName(module.module.getName)
    copyFromBigDL(module, modelBuilder)
    modelBuilder.build
  }

  protected def createRegularizer[T: ClassTag]
    (modelRegularizer : Model.Regularizer)(implicit ev: TensorNumeric[T]): Regularizer[T] = {
    modelRegularizer.getRegularizerType match {
      case RegularizerType.L1Regularizer => L1Regularizer(modelRegularizer.getRegularData(0))
      case RegularizerType.L2Regularizer => L2Regularizer(modelRegularizer.getRegularData(0))
      case RegularizerType.L1L2Regularizer => L1L2Regularizer(modelRegularizer.getRegularData(0),
        modelRegularizer.getRegularData(1))
      case _ => throw new IllegalArgumentException(s"${modelRegularizer.getRegularizerType}" +
        s"cannot be recognized")
    }
  }

  protected def createSerializeRegularizer[T: ClassTag]
    (regularizer : Regularizer[T])(implicit ev: TensorNumeric[T]): Model.Regularizer = {
    val builder = Model.Regularizer.newBuilder
    regularizer match {
      case reg : L1Regularizer[T] =>
        builder.setRegularizerType(RegularizerType.L1Regularizer)
        builder.addRegularData(regularizer.asInstanceOf[L1Regularizer[T]].l1)
      case reg : L2Regularizer[T] =>
        builder.setRegularizerType(RegularizerType.L2Regularizer)
        builder.addRegularData(regularizer.asInstanceOf[L2Regularizer[T]].l2)
      case reg : L1L2Regularizer[T] =>
        builder.setRegularizerType(RegularizerType.L1L2Regularizer)
        val l1l2 = regularizer.asInstanceOf[L1L2Regularizer[T]]
        builder.addRegularData(l1l2.l1)
        builder.addRegularData(l1l2.l2)
    }
    builder.build
  }

  protected def createInitMethod[T: ClassTag]
    (initMethod : Model.InitMethod)(implicit ev: TensorNumeric[T]): InitializationMethod = {
    initMethod match {
      case Model.InitMethod.Default => Default
      case Model.InitMethod.Xavier => Xavier
      case Model.InitMethod.BilinearFiller => BilinearFiller
      case _ => throw new IllegalArgumentException(s"${initMethod}" +
        s"cannot be recognized")
    }
  }

  protected def createSerializeInitMethod[T: ClassTag]
  (initMethod : InitializationMethod)(implicit ev: TensorNumeric[T]): Model.InitMethod = {
    initMethod match {
      case Default => Model.InitMethod.Default
      case Xavier => Model.InitMethod.Xavier
      case  BilinearFiller => Model.InitMethod.BilinearFiller
      case _ => throw new IllegalArgumentException(s"${initMethod}" +
        s"cannot be recognized")
    }
  }
}

case class BigDLModule[T: ClassTag](module : AbstractModule[Activity, Activity, T],
                               tops : Seq[String], bottoms : Seq[String])

object ModelSerializer {

  private val serializerMap = new mutable.HashMap[String, ModelSerializer]()
  private val hdfsPrefix: String = "hdfs:"

  serializerMap("ABS") = AbsSerializer
  serializerMap("ADD") = AddSerializer
  serializerMap("LINEAR") = LinearSerializer
  serializerMap("SEQUENTIAL") = SequentialSerializer
  serializerMap("GRAPH") = GraphSerializer

  case object AbsSerializer extends ModelSerializer {

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

  case object AddSerializer extends ModelSerializer {

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

  case object LinearSerializer extends ModelSerializer {

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

  case object SequentialSerializer extends ModelSerializer {
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

  case object GraphSerializer extends ModelSerializer {
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
        val subModel = serialize(BigDLModule(execution.element, tops, bottoms))
        bigDLModelBuilder.addSubModules(subModel)
      })
      bigDLModelBuilder.setModuleType(ModuleType.GRAPH)
      bigDLModelBuilder.build
    }
  }

  private def load[T: ClassTag](model: BigDLModel)
    (implicit ev: TensorNumeric[T]) : BigDLModule[T] = {
    serializerMap(model.getModuleType.toString).loadModule(model)
  }

  private def serialize[T: ClassTag](bigDLModule : BigDLModule[T])
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

  def loadFromFile[T: ClassTag](modelPath : String)
    (implicit ev: TensorNumeric[T]) : AbstractModule[_, _, _] = {
    val modelBuilder = BigDLModel.newBuilder
    var cis : CodedInputStream = null
    if (modelPath.startsWith(hdfsPrefix)) {
      val byteArrayOut = com.intel.analytics.bigdl.utils.File.readHdfsByte(modelPath)
      cis = CodedInputStream.newInstance(new ByteArrayInputStream(byteArrayOut))
    } else {
      cis = CodedInputStream.newInstance(new FileInputStream(modelPath))
    }
    cis.setSizeLimit(Integer.MAX_VALUE)
    modelBuilder.mergeFrom(cis)
    val bigDLModel = modelBuilder.build()
    load(bigDLModel).module
  }

  def saveToFile[T: ClassTag](modelPath : String, module : AbstractModule[Activity, Activity, T],
    overwrite: Boolean = false)(implicit ev: TensorNumeric[T]) : Unit = {
    val bigDLModule = BigDLModule(module, new ArrayBuffer[String](), new ArrayBuffer[String]())
    val bigDLModel = serialize(bigDLModule)
    if (modelPath.startsWith(hdfsPrefix)) {
      val binaryFile = new Path(modelPath)
      val fs = binaryFile.getFileSystem(new Configuration())
      if (fs.exists(binaryFile)) {
        if (overwrite) {
          fs.delete(binaryFile, true)
        } else {
          throw new RuntimeException(s"file $modelPath already exists")
        }
      }
      val out = fs.create(binaryFile)
      val byteArrayOut = new ByteArrayOutputStream()
      byteArrayOut.write(bigDLModel.toByteArray)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), out, 1024, true)
    } else {
      val binaryFile = new java.io.File(modelPath)
      if (binaryFile.exists()) {
        if (overwrite) {
          binaryFile.delete()
        } else {
          throw new RuntimeException(s"file $modelPath already exists")
        }
      }
      val binaryWriter = new FileOutputStream(binaryFile)
      binaryWriter.write(bigDLModel.toByteArray)
      binaryWriter.close
    }
  }

  def saveModelDefinitionToFile[T: ClassTag](definitionPath : String,
    module : AbstractModule[Activity, Activity, T],
    overwrite: Boolean = false)(implicit ev: TensorNumeric[T]) : Unit = {
    val bigDLModule = BigDLModule(module, new ArrayBuffer[String](), new ArrayBuffer[String]())
    val bigDLModel = serialize(bigDLModule)
    val bigDLModelWithoutWeightsAndBias = BigDLModel.newBuilder(bigDLModel)
    cleantWeightAndBias(bigDLModelWithoutWeightsAndBias)
    val model = bigDLModelWithoutWeightsAndBias.build
    if (definitionPath.startsWith(hdfsPrefix)) {
      val prototxtFile = new Path(definitionPath)
      val fs = prototxtFile.getFileSystem(new Configuration())
      if (fs.exists(prototxtFile)) {
        if (overwrite) {
          fs.delete(prototxtFile, true)
        } else {
          throw new RuntimeException(s"file $definitionPath already exists")
        }
      }
      val out = fs.create(prototxtFile)
      val byteArrayOut = new ByteArrayOutputStream()
      byteArrayOut.write(model.toString.getBytes)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), out, 1024, true)
    } else {
      val prototxtFile = new java.io.File(definitionPath)
      if (prototxtFile.exists()) {
        if (overwrite) {
          prototxtFile.delete()
        } else {
          throw new RuntimeException(s"file $definitionPath already exists")
        }
      }
      val prototxtWriter = new OutputStreamWriter(new FileOutputStream(prototxtFile))
      prototxtWriter.write(model.toString)
      prototxtWriter.close
    }
  }

  private def cleantWeightAndBias(modelBuilder : BigDLModel.Builder): Unit = {
    modelBuilder.clearWeight
    modelBuilder.clearBias
    if (modelBuilder.getSubModulesCount > 0) {
      modelBuilder.clearSubModules
      modelBuilder.getSubModulesList.asScala.foreach(sub => {
        val subModelBuilder = BigDLModel.newBuilder(sub)
        cleantWeightAndBias(subModelBuilder)
        modelBuilder.addSubModules(subModelBuilder.build)
      })
    }
  }
}
