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
package com.intel.analytics.bigdl.utils.serializer

import java.lang.reflect.Field

import com.intel.analytics.bigdl.nn._

import scala.collection.JavaConverters._
import scala.reflect.runtime.universe
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{Tensor, TensorNumericMath}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import serialization.Model.{AttrValue, BigDLModule}
import spire.syntax.field

import scala.collection.mutable
import scala.reflect.ClassTag

object ModuleSerializer extends ModuleSerializable{

  val runtimeMirror = universe.runtimeMirror(getClass.getClassLoader)

  private val moduleMaps = new mutable.HashMap[String, Class[_]]()
  private val classMaps = new mutable.HashMap[Class[_], String]()
  private val deserializerMaps = new mutable.HashMap[String, ModuleSerializable]()
  private val serializerMaps = new mutable.HashMap[Class[_], ModuleSerializable]()

  // generic type definition for type matching

  var tensorNumericType : universe.Type = null
  var tensorType : universe.Type = null
  var regularizerType : universe.Type = null
  var abstractModuleType : universe.Type = null
  var tensorModuleType : universe.Type = null
  var tType : universe.Type = null

  init

  override def loadModule[T: ClassTag](model : BigDLModule)
    (implicit ev: TensorNumeric[T]) : ModuleData[T] = {

    val evidence = scala.reflect.classTag[T]
    val modelAttributes = model.getAttrMap
    val moduleType = model.getModuleType
    val cls = ModuleSerializer.getModuleClsByType(moduleType)
    val constructorMirror = getCostructorMirror(cls)
    val constructorFullParams = constructorMirror.symbol.paramss
    val args = new Array[Object](constructorFullParams(0).size + constructorFullParams(1).size)
    var i = 0;
    constructorFullParams.foreach(map => {
      map.foreach(param => {
        val name = param.name.decodedName.toString
        val ptype = param.typeSignature
        if (ptype.toString == "scala.reflect.ClassTag[T]") {
          args(i) = evidence
        } else if (ptype.toString ==
          tensorNumericType.toString) {
          args(i) = ev
        } else {
          require(modelAttributes.containsKey(name), s"$name value cannot be found")
          val attribute = modelAttributes.get(name)
          val value = DataConverter.getAttributeValue(attribute)
          args(i) = value
        }
        i+= 1
      })
    })
    val module = constructorMirror.apply(args : _*).
      asInstanceOf[AbstractModule[Activity, Activity, T]]
    createBigDLModule(model, module)
  }

  override def serializeModule[T: ClassTag](module : ModuleData[T])
                                           (implicit ev: TensorNumeric[T]) : BigDLModule = {
    val bigDLModelBuilder = BigDLModule.newBuilder
    val cls = module.module.getClass
    val moduleType = getModuleTypeByCls(cls)
    bigDLModelBuilder.setModuleType(moduleType)
   // val constructors = cls.getConstructors()
//    require(constructors.length == 1, "only support one constructor")
   // val constructor = constructors(0)
    val fullParams = getCostructorMirror(cls).symbol.paramss
    val clsTag = scala.reflect.classTag[T]
    val constructorParams = fullParams(0)
    constructorParams.foreach(param => {
      val paramName = param.name.decodedName.toString
      var ptype = param.typeSignature
      val attrBuilder = AttrValue.newBuilder
      // For some modules, fields are declared inside but passed to Super directly
      var field : Field = null
      try {
        field = cls.getDeclaredField(paramName)
      } catch {
        case e : NoSuchFieldException =>
          field = cls.getSuperclass.getDeclaredField(paramName)
      }
      field.setAccessible(true)
      val fieldValue = field.get(module.module)
      DataConverter.setAttributeValue(attrBuilder, fieldValue, ptype)
      bigDLModelBuilder.putAttr(paramName, attrBuilder.build)
    })
    copyFromBigDL(module, bigDLModelBuilder)
    createSerializeBigDLModule(bigDLModelBuilder, module)
  }


  def serialize[T: ClassTag](bigDLModule : ModuleData[T])
                            (implicit ev: TensorNumeric[T])
    : BigDLModule = {
    val module = bigDLModule.module
    val cls = module.getClass
    serializerMaps(cls).serializeModule(bigDLModule)
  }

  def load[T: ClassTag](model: BigDLModule)
                       (implicit ev: TensorNumeric[T]) : ModuleData[T] = {
    deserializerMaps(model.getModuleType).loadModule(model)
  }



  def registerModule(moduleType : String, moduleCls : Class[_],
    serializer : ModuleSerializable) : Unit = {
    moduleMaps(moduleType) = moduleCls
    classMaps(moduleCls) = moduleType
    serializerMaps(moduleCls) = serializer
    deserializerMaps(moduleType) = serializer
  }

  def getModuleClsByType(moduleType : String) : Class[_] = {
    require(moduleMaps.contains(moduleType), s"$moduleType is not supported")
    moduleMaps(moduleType)
  }

  def getModuleTypeByCls(cls : Class[_]) : String = {
    require(classMaps.contains(cls), s"$cls is not supported")
    classMaps(cls)
  }

  def getCostructorMirror[T : ClassTag](cls : Class[_]) : universe.MethodMirror = {

    val clsSymbol = runtimeMirror.classSymbol(cls)
    val cm = runtimeMirror.reflectClass(clsSymbol)
    // to make it compatible with both 2.11 and 2.10
    // val ctorC = clsSymbol.toType.declaration(universe.nme.CONSTRUCTOR).asMethod
    val ctorCs = clsSymbol.toType.declaration(universe.nme.CONSTRUCTOR)
    val primary : Option[universe.MethodSymbol] = ctorCs.asTerm.alternatives.collectFirst{
      case cstor : universe.MethodSymbol if cstor.isPrimaryConstructor => cstor
    }
    // cm.reflectConstructor(ctorC)
    cm.reflectConstructor(primary.get)
  }

  private def init() : Unit = {
    registerAllModules
    initializeDeclaredTypes
  }

  private def registerAllModules : Unit = {
    registerModule("Abs", Class.forName("com.intel.analytics.bigdl.nn.Abs"), Abs)
    registerModule("Add", Class.forName("com.intel.analytics.bigdl.nn.Add"), Add)
    registerModule("AddConstant", Class.forName("com.intel.analytics.bigdl.nn.AddConstant"),
      AddConstant)
    registerModule("BatchNormalization",
      Class.forName("com.intel.analytics.bigdl.nn.BatchNormalization"), BatchNormalization)
    registerModule("Bilinear", Class.forName("com.intel.analytics.bigdl.nn.Bilinear"), Bilinear)
    registerModule("BiRecurrent", Class.forName("com.intel.analytics.bigdl.nn.BiRecurrent"),
      BiRecurrent)
    registerModule("Bottle", Class.forName("com.intel.analytics.bigdl.nn.Bottle"), Bottle)
    registerModule("CAdd", Class.forName("com.intel.analytics.bigdl.nn.CAdd"), CAdd)
    registerModule("CAddTable", Class.forName("com.intel.analytics.bigdl.nn.CAddTable"), CAddTable)
    registerModule("CDivTable", Class.forName("com.intel.analytics.bigdl.nn.CDivTable"), CDivTable)
    registerModule("Clamp", Class.forName("com.intel.analytics.bigdl.nn.Clamp"), Clamp)
    registerModule("CMaxTable", Class.forName("com.intel.analytics.bigdl.nn.CMaxTable"), CMaxTable)
    registerModule("CMinTable", Class.forName("com.intel.analytics.bigdl.nn.CMinTable"), CMinTable)
    registerModule("CMul", Class.forName("com.intel.analytics.bigdl.nn.CMul"), CMul)
    registerModule("CMulTable", Class.forName("com.intel.analytics.bigdl.nn.CMulTable"), CMulTable)
    registerModule("Concat", Class.forName("com.intel.analytics.bigdl.nn.Concat"), Concat)
    registerModule("ConcatTable", Class.forName("com.intel.analytics.bigdl.nn.ConcatTable"),
      ConcatTable)
    registerModule("Contiguous", Class.forName("com.intel.analytics.bigdl.nn.Contiguous"),
      Contiguous)
    registerModule("Cosine", Class.forName("com.intel.analytics.bigdl.nn.Cosine"),
      Cosine)
    registerModule("CosineDistance", Class.forName("com.intel.analytics.bigdl.nn.CosineDistance"),
      CosineDistance)
    registerModule("CSubTable", Class.forName("com.intel.analytics.bigdl.nn.CSubTable"),
      CSubTable)
    registerModule("DotProduct", Class.forName("com.intel.analytics.bigdl.nn.DotProduct"),
      DotProduct)
    registerModule("Dropout", Class.forName("com.intel.analytics.bigdl.nn.Dropout"),
      Dropout)
    registerModule("Echo", Class.forName("com.intel.analytics.bigdl.nn.Echo"),
      Echo)
    registerModule("ELU", Class.forName("com.intel.analytics.bigdl.nn.ELU"),
      ELU)
    registerModule("Euclidean", Class.forName("com.intel.analytics.bigdl.nn.Euclidean"),
      Euclidean)
    registerModule("Exp", Class.forName("com.intel.analytics.bigdl.nn.Exp"),
      Exp)
    registerModule("FlattenTable", Class.forName("com.intel.analytics.bigdl.nn.FlattenTable"),
      FlattenTable)
    registerModule("GradientReversal", Class.forName
    ("com.intel.analytics.bigdl.nn.GradientReversal"), GradientReversal)
    registerModule("Graph", Class.forName("com.intel.analytics.bigdl.nn.Graph"), Graph)
    registerModule("GRU", Class.forName("com.intel.analytics.bigdl.nn.GRU"), GRU)
    registerModule("HardShrink", Class.forName("com.intel.analytics.bigdl.nn.HardShrink"),
      HardShrink)
    registerModule("HardTanh", Class.forName("com.intel.analytics.bigdl.nn.HardTanh"), HardTanh)
    registerModule("Identity", Class.forName("com.intel.analytics.bigdl.nn.Identity"), Identity)
    registerModule("Index", Class.forName("com.intel.analytics.bigdl.nn.Index"), Index)
    registerModule("InferReshape", Class.forName("com.intel.analytics.bigdl.nn.InferReshape"),
      InferReshape)
    registerModule("JoinTable", Class.forName("com.intel.analytics.bigdl.nn.JoinTable"), JoinTable)
    registerModule("L1Penalty", Class.forName("com.intel.analytics.bigdl.nn.L1Penalty"), L1Penalty)
    registerModule("LeakyReLU", Class.forName("com.intel.analytics.bigdl.nn.LeakyReLU"), LeakyReLU)
    registerModule("Linear", Class.forName("com.intel.analytics.bigdl.nn.Linear"), Linear)
    registerModule("Log", Class.forName("com.intel.analytics.bigdl.nn.Log"), Log)
    registerModule("LogSigmoid", Class.forName("com.intel.analytics.bigdl.nn.LogSigmoid"),
      LogSigmoid)
    registerModule("LogSoftMax", Class.forName("com.intel.analytics.bigdl.nn.LogSoftMax"),
      LogSoftMax)
    registerModule("LookupTable", Class.forName("com.intel.analytics.bigdl.nn.LookupTable"),
      LookupTable)
    registerModule("LSTM", Class.forName("com.intel.analytics.bigdl.nn.LSTM"), LSTM)
    registerModule("LSTMPeephole", Class.forName("com.intel.analytics.bigdl.nn.LSTMPeephole"),
      LSTMPeephole)
    registerModule("MapTable", Class.forName("com.intel.analytics.bigdl.nn.MapTable"), MapTable)
    registerModule("MaskedSelect", Class.forName("com.intel.analytics.bigdl.nn.MaskedSelect"),
      MaskedSelect)
    registerModule("Max", Class.forName("com.intel.analytics.bigdl.nn.Max"), Max)
    registerModule("Mean", Class.forName("com.intel.analytics.bigdl.nn.Mean"), Mean)
    registerModule("Min", Class.forName("com.intel.analytics.bigdl.nn.Min"), Min)
    registerModule("MixtureTable", Class.forName("com.intel.analytics.bigdl.nn.MixtureTable"),
      MixtureTable)
    registerModule("MM", Class.forName("com.intel.analytics.bigdl.nn.MM"), MM)
    registerModule("Mul", Class.forName("com.intel.analytics.bigdl.nn.Mul"), Mul)
    registerModule("MulConstant", Class.forName("com.intel.analytics.bigdl.nn.MulConstant"),
      MulConstant)
    registerModule("MV", Class.forName("com.intel.analytics.bigdl.nn.MV"), MV)
    registerModule("Narrow", Class.forName("com.intel.analytics.bigdl.nn.Narrow"), Narrow)
    registerModule("NarrowTable", Class.forName("com.intel.analytics.bigdl.nn.NarrowTable"),
      NarrowTable)
    registerModule("Normalize", Class.forName("com.intel.analytics.bigdl.nn.Normalize"),
      Normalize)
    registerModule("Pack", Class.forName("com.intel.analytics.bigdl.nn.Pack"), Pack)
    registerModule("Padding", Class.forName("com.intel.analytics.bigdl.nn.Padding"), Padding)
    registerModule("PairwiseDistance",
      Class.forName("com.intel.analytics.bigdl.nn.PairwiseDistance"), PairwiseDistance)
    registerModule("ParallelTable", Class.forName("com.intel.analytics.bigdl.nn.ParallelTable"),
      ParallelTable)
    registerModule("Power", Class.forName("com.intel.analytics.bigdl.nn.Power"), Power)
    registerModule("PReLU", Class.forName("com.intel.analytics.bigdl.nn.PReLU"), PReLU)
    registerModule("Recurrent", Class.forName("com.intel.analytics.bigdl.nn.Recurrent"), Recurrent)
    registerModule("ReLU", Class.forName("com.intel.analytics.bigdl.nn.ReLU"), ReLU)
    registerModule("ReLU6", Class.forName("com.intel.analytics.bigdl.nn.ReLU6"), ReLU6)
    registerModule("Replicate", Class.forName("com.intel.analytics.bigdl.nn.Replicate"), Replicate)
    registerModule("Reshape", Class.forName("com.intel.analytics.bigdl.nn.Reshape"), Reshape)
    registerModule("Reverse", Class.forName("com.intel.analytics.bigdl.nn.Reverse"), Reverse)
    registerModule("RnnCell", Class.forName("com.intel.analytics.bigdl.nn.RnnCell"), RnnCell)
    registerModule("RoiPooling", Class.forName("com.intel.analytics.bigdl.nn.RoiPooling"),
      RoiPooling)
    registerModule("RReLU", Class.forName("com.intel.analytics.bigdl.nn.RReLU"), RReLU)
    registerModule("Scale", Class.forName("com.intel.analytics.bigdl.nn.Scale"), Scale)
    registerModule("Select", Class.forName("com.intel.analytics.bigdl.nn.Select"), Select)
    registerModule("SelectTable", Class.forName("com.intel.analytics.bigdl.nn.SelectTable"),
      SelectTable)
    registerModule("Sequential", Class.forName("com.intel.analytics.bigdl.nn.Sequential"),
      Sequential)
    registerModule("Sigmoid", Class.forName("com.intel.analytics.bigdl.nn.Sigmoid"), Sigmoid)
    registerModule("SoftMax", Class.forName("com.intel.analytics.bigdl.nn.SoftMax"), SoftMax)
    registerModule("SoftMin", Class.forName("com.intel.analytics.bigdl.nn.SoftMin"), SoftMin)
    registerModule("SoftPlus", Class.forName("com.intel.analytics.bigdl.nn.SoftPlus"), SoftPlus)
    registerModule("SoftShrink", Class.forName("com.intel.analytics.bigdl.nn.SoftShrink"),
      SoftShrink)
    registerModule("SoftSign", Class.forName("com.intel.analytics.bigdl.nn.SoftSign"),
      SoftSign)
    registerModule("SpatialAveragePooling",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialAveragePooling"), SpatialAveragePooling)
    registerModule("SpatialBatchNormalization",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialBatchNormalization"),
      SpatialBatchNormalization)
    registerModule("SpatialContrastiveNormalization",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialContrastiveNormalization"),
      SpatialContrastiveNormalization)
    registerModule("SpatialConvolution",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialConvolution"), SpatialConvolution)
    registerModule("SpatialConvolutionMap",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialConvolutionMap"), SpatialConvolutionMap)
    registerModule("SpatialCrossMapLRN",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialCrossMapLRN"), SpatialCrossMapLRN)
    registerModule("SpatialDilatedConvolution",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialDilatedConvolution"),
      SpatialDilatedConvolution)
    registerModule("SpatialDivisiveNormalization",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialDivisiveNormalization"),
      SpatialDivisiveNormalization)
    registerModule("SpatialFullConvolution",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialFullConvolution"),
      SpatialFullConvolution)
    registerModule("SpatialMaxPooling",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialMaxPooling"), SpatialMaxPooling)
    registerModule("SpatialShareConvolution",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialShareConvolution"),
      SpatialShareConvolution)
    registerModule("SpatialSubtractiveNormalization",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialSubtractiveNormalization"),
      SpatialSubtractiveNormalization)
    registerModule("SpatialZeroPadding",
      Class.forName("com.intel.analytics.bigdl.nn.SpatialZeroPadding"), SpatialZeroPadding)
    registerModule("SplitTable", Class.forName("com.intel.analytics.bigdl.nn.SplitTable"),
      SplitTable)
    registerModule("Sqrt", Class.forName("com.intel.analytics.bigdl.nn.Sqrt"), Sqrt)
    registerModule("Square", Class.forName("com.intel.analytics.bigdl.nn.Square"), Square)
    registerModule("Squeeze", Class.forName("com.intel.analytics.bigdl.nn.Squeeze"), Squeeze)
    registerModule("Sum", Class.forName("com.intel.analytics.bigdl.nn.Sum"), Sum)
    registerModule("Tanh", Class.forName("com.intel.analytics.bigdl.nn.Tanh"), Tanh)
    registerModule("TanhShrink", Class.forName("com.intel.analytics.bigdl.nn.TanhShrink"),
      TanhShrink)
    registerModule("Threshold", Class.forName("com.intel.analytics.bigdl.nn.Threshold"), Threshold)
    registerModule("TimeDistributed", Class.forName("com.intel.analytics.bigdl.nn.TimeDistributed"),
      TimeDistributed)
    registerModule("Transpose", Class.forName("com.intel.analytics.bigdl.nn.Transpose"), Transpose)
    registerModule("Unsqueeze", Class.forName("com.intel.analytics.bigdl.nn.Unsqueeze"), Unsqueeze)
    registerModule("View", Class.forName("com.intel.analytics.bigdl.nn.View"), View)
    registerModule("VolumetricConvolution",
      Class.forName("com.intel.analytics.bigdl.nn.VolumetricConvolution"), VolumetricConvolution)
    registerModule("VolumetricMaxPooling",
      Class.forName("com.intel.analytics.bigdl.nn.VolumetricMaxPooling"), VolumetricMaxPooling)

  }

  private def initializeDeclaredTypes() : Unit = {

    /*
    val tensorNumericCls = Class.
      forName("com.intel.analytics.bigdl.tensor.TensorNumericMath$TensorNumeric")
    tensorNumericType = runtimeMirror.
      classSymbol(tensorNumericCls).selfType

    val tensorCls = Class.forName("com.intel.analytics.bigdl.tensor.Tensor")
    tensorType = runtimeMirror.
      classSymbol(tensorCls).selfType

    val regularizerCls = Class.forName("com.intel.analytics.bigdl.optim.Regularizer")
    regularizerType = runtimeMirror.
      classSymbol(regularizerCls).selfType

    val abstractModuleCls = Class.forName("com.intel.analytics.bigdl.nn.abstractnn.AbstractModule")
    abstractModuleType = runtimeMirror.classSymbol(abstractModuleCls).selfType

    val tensorModuleCls = Class.forName("com.intel.analytics.bigdl.nn.abstractnn.TensorModule")
    tensorModuleType = runtimeMirror.classSymbol(tensorModuleCls).selfType
    */
    var wrapperCls = Class.forName("com.intel.analytics.bigdl.utils.serializer.GenericTypeWrapper")
    val fullParams = getCostructorMirror(wrapperCls).symbol.paramss
    fullParams.foreach(map => {
      map.foreach(param => {
        val name = param.name.decodedName.toString
        val ptype = param.typeSignature
        if (name == "tensor") {
          tensorType = ptype
        } else if (name == "regularizer") {
          regularizerType = ptype
        } else if (name == "abstractModule") {
          abstractModuleType = ptype
        } else if (name == "tensorModule") {
          tensorModuleType = ptype
        } else if (name == "ev") {
          tensorNumericType = ptype
        } else if (name == "ttpe") {
          tType = ptype
        }
      })
    })
  }
}

private class GenericTypeWrapper[T: ClassTag](tensor : Tensor[T],
                                              regularizer : Regularizer[T],
                                              abstractModule: AbstractModule[Activity, Activity, T],
                                              tensorModule : TensorModule[T],
                                              ttpe : T
                                             )
(implicit ev: TensorNumeric[T]) {

}

