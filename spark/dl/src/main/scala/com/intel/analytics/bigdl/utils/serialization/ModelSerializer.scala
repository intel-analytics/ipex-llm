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
import com.intel.analytics.bigdl.utils.serialization.ModelSerializer.AddSerializer._
import com.intel.analytics.bigdl.utils.serialization.ModelSerializer.BottleSerializer._
import com.intel.analytics.bigdl.utils.serialization.ModelSerializer.LinearSerializer._
import serialization.Model
import serialization.Model.BigDLModel.ModuleType
import serialization.Model._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

private[serialization] object ModelSerializer {

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

  case object  AddConstantSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val addConstParam = model.getAddConstParam
      val constScalar = addConstParam.getConstScalar
      val inPlace = if (addConstParam.hasInPlace) addConstParam.getInPlace else false
      val addConst = AddConstant[T](constScalar, inPlace)
      createBigDLModule(model, addConst.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val addConst = module.module.asInstanceOf[AddConstant[T]]
      val constantScalar = addConst.constant_scalar
      val inPlace = addConst.inplace
      val addParam = AddConstParam.newBuilder
      addParam.setConstScalar(constantScalar)
      addParam.setInPlace(inPlace)
      bigDLModelBuilder.setModuleType(ModuleType.ADDCONST)
      bigDLModelBuilder.setAddConstParam(addParam.build)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }

  }

  case object  BatchNormSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val batchNormParam = model.getBatchNormParam
      val nOutput = batchNormParam.getNOutput
      val eps = if (batchNormParam.hasEps) batchNormParam.getEps else 1e-5
      val momentum = if (batchNormParam.hasMomentum) batchNormParam.getMomentum else 0.1
      val affine = if (batchNormParam.hasAffine) batchNormParam.getAffine else true
      val batchNorm = BatchNormalization[T](nOutput, eps, momentum, affine)
      createBigDLModule(model, batchNorm.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val batchNorm = module.module.asInstanceOf[BatchNormalization[T]]
      val nOutPut = batchNorm.nOutput
      val eps = batchNorm.eps
      val momentum = batchNorm.momentum
      val affine = batchNorm.affine
      val batchNormParam = BatchNormParam.newBuilder
      batchNormParam.setNOutput(nOutPut)
      batchNormParam.setEps(eps)
      batchNormParam.setMomentum(momentum)
      batchNormParam.setAffine(affine)
      bigDLModelBuilder.setModuleType(ModuleType.BATCHNORM)
      bigDLModelBuilder.setBatchNormParam(batchNormParam.build)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }

  }

  case object  BiLinearSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      createBigDLModule(model, createBiLinear(model).
        asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    private def createBiLinear[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T]):
      Bilinear[T] = {
      val biLinearParams = model.getBiLinearParam
      val inputSize1 = biLinearParams.getInputSize1
      val inputSize2 = biLinearParams.getInputSize2
      val outputSize = biLinearParams.getOutputSize
      val biasRes = if (biLinearParams.hasBiasRes) biLinearParams.getBiasRes else true
      var wRegularizer : Regularizer[T] = null
      if (biLinearParams.hasWRegularizer) {
        wRegularizer = createRegularizer(biLinearParams.getWRegularizer)
      }
      var bRegularizer : Regularizer[T] = null
      if (biLinearParams.hasBRegularizer) {
        bRegularizer = createRegularizer(biLinearParams.getBRegularizer)
      }
      Bilinear[T](inputSize1, inputSize2, outputSize, biasRes, wRegularizer, bRegularizer)
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val biLinear = module.module.asInstanceOf[Bilinear[T]]
      createSerializeBiLinear(bigDLModelBuilder, biLinear)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }

    private def createSerializeBiLinear[T: ClassTag](
      modelBuilder : BigDLModel.Builder, biLinear : Bilinear[T])
      (implicit ev: TensorNumeric[T]): Unit = {
      val linearParam = BiLinearParam.newBuilder
      linearParam.setInputSize1(biLinear.inputSize1)
      linearParam.setInputSize2(biLinear.inputSize2)
      linearParam.setOutputSize(biLinear.outputSize)
      linearParam.setBiasRes(biLinear.biasRes)
      if (biLinear.wRegularizer != null) {
        linearParam.setWRegularizer(createSerializeRegularizer(biLinear.wRegularizer))
      }
      if (biLinear.bRegularizer != null) {
        linearParam.setBRegularizer(createSerializeRegularizer(biLinear.bRegularizer))
      }
      modelBuilder.setBiLinearParam(linearParam.build)
      modelBuilder.setModuleType(ModuleType.BILINEAR)
    }

  }

  case object BiRecurrentSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val birecurrentParam = model.getBiRecurrentParam
      val merge = if (birecurrentParam.hasMerge) birecurrentParam.getMerge else null
      var mergeModule : AbstractModule[Table, Tensor[T], T] = null
      if (merge != null) {
        mergeModule = loadModule(merge).module.
          asInstanceOf[AbstractModule[Table, Tensor[T], T]]
      } else {
        mergeModule = CAddTable[T]()
      }
      createBigDLModule(model, BiRecurrent[T](mergeModule).
        asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val birecurrentParam = BiRecurrentParam.newBuilder
      val mergeModel : BigDLModel = serialize(BigDLModule(module.module, new ArrayBuffer[String](),
        new ArrayBuffer[String]()))
      birecurrentParam.setMerge(mergeModel)
      bigDLModelBuilder.setBiRecurrentParam(birecurrentParam.build)
      bigDLModelBuilder.setModuleType(ModuleType.BIRECURRENT)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object BottleSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val subModules = model.getSubModulesList.asScala
      require(subModules.size == 1, "bottle sub module size should be 1")
      val module = load(subModules(0)).module
      val bottleParam = model.getBottleParam
      val nInputDim = bottleParam.getNInputDim
      val nOutputDim = bottleParam.getNOutputDim1
      val bottle = Bottle[T](module, nInputDim, nOutputDim)
      createBigDLModule(model, bottle.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val bottle = module.module.asInstanceOf[Bottle[T]]
      require(bottle.modules.size == 1, "bottle sub module size should be 1")
      val model = serialize(BigDLModule(bottle.modules(0), module.tops, module.bottoms))
      val bottleParam = BottleParam.newBuilder
      bottleParam.setNInputDim(bottle.nInputDim)
      bottleParam.setNOutputDim1(bottle.nOutputDim1)
      bottleParam.setModule(model)
      bigDLModelBuilder.addSubModules(model)
      bigDLModelBuilder.setBottleParam(bottleParam.build)
      bigDLModelBuilder.setModuleType(ModuleType.BOTTLE)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object CaddSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val caddParam = model.getCaddParam
      val size = caddParam.getSizeList.asScala
      val caddSize = Array[Int](size.size)
      (1 until size.size).foreach(i => caddSize(i - 1) = size(i -1))
      createBigDLModule(model, CAdd[T](caddSize).
        asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val caddParam = CaddParam.newBuilder
      module.module.asInstanceOf[CAdd[Double]].size.foreach(size => caddParam.addSize(size))
      bigDLModelBuilder.setCaddParam(caddParam.build)
      bigDLModelBuilder.setModuleType(ModuleType.CADD)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }


  case object CaddTableSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val caddTableParam = model.getCAddTableParam
      val inPlace = if (caddTableParam.hasInplace) caddTableParam.getInplace else false
      createBigDLModule(model, CAddTable[T](inPlace).
        asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val caddTableParam = CaddTableParam.newBuilder
      caddTableParam.setInplace(module.module.asInstanceOf[CAddTable[T]].inplace)
      bigDLModelBuilder.setCAddTableParam(caddTableParam.build)
      bigDLModelBuilder.setModuleType(ModuleType.CADDTABLE)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object CDivTableSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val cdiveTable = new CDivTable[T]()
      createBigDLModule(model, cdiveTable.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      bigDLModelBuilder.setModuleType(ModuleType.CDIVTABLE)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object ClampSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val clampParam = model.getClampParam
      val clamp = new Clamp[T](clampParam.getMin, clampParam.getMax)
      createBigDLModule(model, clamp.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val clampParam = ClampParam.newBuilder
      val clamp = module.module.asInstanceOf[Clamp[T]]
      clampParam.setMin(clamp.minV)
      clampParam.setMax(clamp.maxV)
      bigDLModelBuilder.setClampParam(clampParam.build)
      bigDLModelBuilder.setModuleType(ModuleType.CLAMP)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object CMaxTableSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val cmaxTable = new CMaxTable[T]()
      createBigDLModule(model, cmaxTable.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      bigDLModelBuilder.setModuleType(ModuleType.CMAXTABLE)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object CMinTableSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val cminTable = new CMinTable[T]()
      createBigDLModule(model, cminTable.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      bigDLModelBuilder.setModuleType(ModuleType.CMINTABLE)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object CMulSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val cMulSize = model.getCmulParam.getSizeList.asScala
      val size = Array[Int](cMulSize.size)
      (1 until cMulSize.size).foreach(i => size(i - 1) = cMulSize(i - 1))
      val cmul = new CMul[T](size)
      createBigDLModule(model, cmul.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val size = module.module.asInstanceOf[CMul[T]].size
      val cmulParam = CMulParam.newBuilder
      size.foreach(i => cmulParam.addSize(i))
      bigDLModelBuilder.setCmulParam(cmulParam.build)
      bigDLModelBuilder.setModuleType(ModuleType.CMUL)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object CMulTableSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      createBigDLModule(model, CMulTable[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      bigDLModelBuilder.setModuleType(ModuleType.CMULTABLE)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object ConcatSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val dimension = model.getConcatParam.getDimension
      val concat = Concat[T](dimension)
      model.getSubModulesList.asScala.foreach(subModule => {
        concat.add(load(subModule).module
          .asInstanceOf[AbstractModule[Activity, Activity, T]])
      })
      createBigDLModule(model, concat.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val concat = module.module.asInstanceOf[Concat[T]]
      concat.modules.foreach(subModule => {
        val subModel = serialize(BigDLModule(subModule, module.tops, module.bottoms))
        bigDLModelBuilder.addSubModules(subModel)
      })
      val concatParam = ConcatParam.newBuilder
      concatParam.setDimension(concat.dimension)
      bigDLModelBuilder.setConcatParam(concatParam.build)
      bigDLModelBuilder.setModuleType(ModuleType.CONCAT)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object ConcatTableSerializer extends AbstractModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val subModules = model.getSubModulesList.asScala
      val concatTable = new ConcatTable[T]()
      subModules.foreach(subModule => {
        val bigDLModule = load(subModule)
        concatTable.add(bigDLModule.module.
          asInstanceOf[AbstractModule[_ <: Activity, _ <: Activity, T]])
      })
      createBigDLModule(model, concatTable.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val concatTable = module.module.asInstanceOf[ConcatTable[T]]
      concatTable.modules.foreach(subModule => {
        val subModel = serialize(BigDLModule(subModule, module.tops, module.bottoms))
        bigDLModelBuilder.addSubModules(subModel)
      })
      bigDLModelBuilder.setModuleType(ModuleType.CONCATTABLE)
      bigDLModelBuilder.build
    }
  }

  case object ContiguousSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      createBigDLModule(model, Contiguous[T]().asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      bigDLModelBuilder.setModuleType(ModuleType.CONTIGUOUS)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object CosineSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val cosineParam = model.getCosineParam
      createBigDLModule(model, Cosine[T](cosineParam.getInputSize,
        cosineParam.getOutputSize).asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      val cosineParam = CosineParam.newBuilder
      val cosine = module.module.asInstanceOf[Cosine[T]]
      cosineParam.setInputSize(cosine.inputSize)
      cosineParam.setOutputSize(cosine.outputSize)
      bigDLModelBuilder.setCosineParam(cosineParam.build)
      bigDLModelBuilder.setModuleType(ModuleType.COSINE)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object CosineDistanceSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      createBigDLModule(model, CosineDistance[T]().
        asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      bigDLModelBuilder.setModuleType(ModuleType.COSINEDISTANCE)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object CSubTableDistanceSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      createBigDLModule(model, CSubTable[T]().
        asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      bigDLModelBuilder.setModuleType(ModuleType.CSUBTABLE)
      createSerializeBigDLModule(bigDLModelBuilder, module)
    }
  }

  case object DotProductDistanceSerializer extends AbstractModelSerializer {

    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      createBigDLModule(model, DotProduct[T]().
        asInstanceOf[AbstractModule[Activity, Activity, T]])
    }

    override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                             (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      bigDLModelBuilder.setModuleType(ModuleType.DOTPRODUCT)
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
      case addConst : AddConstant[_] => AddConstantSerializer.serializeModule(bigDLModule)
      case batchNorm : BatchNormalization[_] => BatchNormSerializer.serializeModule(bigDLModule)
      case biLinear : Bilinear[_] => BiLinearSerializer.serializeModule(bigDLModule)
      case biRecurrent : BiRecurrent[_] => BiRecurrentSerializer.serializeModule(bigDLModule)
      case bottle : Bottle[_] => BottleSerializer.serializeModule(bigDLModule)
      case cadd : CAdd[_] => CaddSerializer.serializeModule(bigDLModule)
      case caddTable : CAddTable[_] => CaddTableSerializer.serializeModule(bigDLModule)
      case concatTable : ConcatTable[_] => ConcatTableSerializer.serializeModule(bigDLModule)
      case cdivTable : CDivTable[_] => CDivTableSerializer.serializeModule(bigDLModule)
      case clamp : Clamp[_] => ClampSerializer.serializeModule(bigDLModule)
      case cmaxTable : CMaxTable[_] => CMaxTableSerializer.serializeModule(bigDLModule)
      case cminTable : CMinTable[_] => CMinTableSerializer.serializeModule(bigDLModule)
      case cmul : CMul[_] => CMulSerializer.serializeModule(bigDLModule)
      case cmulTable : CMulTable[_] => CMulTableSerializer.serializeModule(bigDLModule)
      case concat : Concat[_] => ConcatSerializer.serializeModule(bigDLModule)
      case contiguous : Contiguous[_] => ContiguousSerializer.serializeModule(bigDLModule)
      case cosine : Cosine[_] => CosineSerializer.serializeModule(bigDLModule)
      case cosineDis : CosineDistance[_] => CosineDistanceSerializer.serializeModule(bigDLModule)
      case csubTable : CSubTable[_] => CSubTableDistanceSerializer.serializeModule(bigDLModule)
      case dotProduct : DotProduct[_] => DotProductDistanceSerializer.serializeModule(bigDLModule)
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
    serializerMap("ADDCONST") = AddConstantSerializer
    serializerMap("BATCHNORM") = BatchNormSerializer
    serializerMap("BILINEAR") = BiLinearSerializer
    serializerMap("BIRECURRENT") = BiRecurrentSerializer
    serializerMap("BOTTLE") = BottleSerializer
    serializerMap("CADD") = CaddSerializer
    serializerMap("CADDTABLE") = CaddTableSerializer
    serializerMap("CONCATTABLE") = ConcatTableSerializer
    serializerMap("CDIVTABLE") = CDivTableSerializer
    serializerMap("CLAMP") = ClampSerializer
    serializerMap("CMAXTABLE") = CMaxTableSerializer
    serializerMap("CMINTABLE") = CMinTableSerializer
    serializerMap("CMULTABLE") = CMulTableSerializer
    serializerMap("CMUL") = CMulSerializer
    serializerMap("CONCAT") = ConcatSerializer
    serializerMap("CONTIGUOUS") = ContiguousSerializer
    serializerMap("COSINE") = CosineSerializer
    serializerMap("COSINEDISTANCE") = CosineDistanceSerializer
    serializerMap("CSUBTABLE") = CSubTableDistanceSerializer
    serializerMap("DOTPRODUCT") = DotProductDistanceSerializer
    serializerMap("LINEAR") = LinearSerializer
    serializerMap("SEQUENTIAL") = SequentialSerializer
    serializerMap("GRAPH") = GraphSerializer
  }

}
