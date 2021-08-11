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

import java.io._
import java.nio.ByteBuffer
import java.security.{DigestInputStream, DigestOutputStream, MessageDigest}

import scala.collection.JavaConverters._
import com.google.protobuf.CodedInputStream
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{DenseType, QuantizedTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.converters.TensorConverter
import com.intel.analytics.bigdl.utils.serializer.converters.DataReaderWriter
import com.intel.analytics.bigdl.utils.{File, FileReader, FileWriter, Table}
import com.intel.analytics.bigdl.serialization.Bigdl._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object ModuleLoader {

  /**
   * load module from `modelPath`
   * @param modelPath  path where protobuf formatted module is stored
   * @param weightPath optional : weight path
   * @param ev numeric ops
   * @tparam T data type
   * @return loaded BigDL module
   */
  def loadFromFile[T: ClassTag](modelPath : String, weightPath : String = null)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val modelBuilder = BigDLModule.newBuilder
    val inputBytes = File.readBytes(modelPath)
    var cis : CodedInputStream = CodedInputStream.newInstance(new ByteArrayInputStream(inputBytes))
    cis.setSizeLimit(Integer.MAX_VALUE)
    modelBuilder.mergeFrom(cis)
    val bigDLModel = modelBuilder.build()
    val storages = new mutable.HashMap[Int, Any]()
    var deserializationContext : DeserializeContext = null
    if (weightPath == null) {
      deserializationContext = DeserializeContext(bigDLModel, storages, ProtoStorageType)
      initTensorStorage(deserializationContext)
    } else {
      deserializationContext = DeserializeContext(bigDLModel, storages, BigDLStorage)
      initTensorStorage(deserializationContext, weightPath)
    }
    ModuleSerializer.load(deserializationContext).module
  }

  private def initTensorStorage[T: ClassTag](context: DeserializeContext, weightPath : String)
                                            (implicit ev: TensorNumeric[T]): Unit = {
    val magicNo = SerConst.MAGIC_NO
    var fr: FileReader = null
    var in: InputStream = null
    var objFile: ObjectInputStream = null
    val storages = context.storages
    try {
      fr = FileReader(weightPath)
      in = fr.open()
      val digest = MessageDigest.getInstance(SerConst.DIGEST_TYPE)
      val digestInputStream = new DigestInputStream(in, digest)
      val dataInputStream = new DataInputStream(digestInputStream)
      digestInputStream.on(true)
      val magicNumber = dataInputStream.readInt
      require(magicNumber == magicNo,
        s"Magic number mismatch, expected $magicNo, actual $magicNumber")

      val totalCount = dataInputStream.readInt
      // Read each storage data and convert to storage
      for (i <- 0 until totalCount) {
        val storageId = dataInputStream.readInt
        val dataType = BigDLDataType(dataInputStream.readInt)
        val reader = DataReaderWriter(dataType)
        val size = dataInputStream.readInt
        val data = reader.read(dataInputStream, size)
        storages(storageId) = data
      }
      digestInputStream.on(false)

      val digestLen = dataInputStream.readInt

      val storedDigest = new Array[Byte](digestLen)

      dataInputStream.read(storedDigest)

      val calculatedDigest = digestInputStream.getMessageDigest.digest

      require(calculatedDigest.length == digestLen, "checksum error, size mismatch")

      for (i <- 0 until digestLen) {
        require(calculatedDigest(i) == storedDigest(i), "check sum error, please check weight file")
      }

    } finally {
      if (null != in) in.close()
      if (null != fr) fr.close()
      if (null != objFile) objFile.close()
    }
  }

  private[bigdl] def initTensorStorage[T: ClassTag](context: DeserializeContext)
                                            (implicit ev: TensorNumeric[T]): Unit = {
    val attrMap = context.bigdlModule.getAttrMap

    val storagesMap = attrMap.get(SerConst.GLOBAL_STORAGE).getNameAttrListValue.getAttrMap

    storagesMap.asScala.foreach(map => {
      val storages = context.storages
      val tensorId = map._1.toInt
      val tensorValue = map._2.getTensorValue
      val storageId = tensorValue.getStorage.getId
      val tensor = TensorConverter.getAttributeValue(context, map._2).asInstanceOf[Tensor[_]]
      val tensorStorage = tensorValue.getTensorType match {
        case TensorType.DENSE => tensor.storage()
        case TensorType.QUANT => tensor.asInstanceOf[QuantizedTensor[_]].getStorage
        case _ => throw new UnsupportedOperationException("Unsupported Tensor Type")
      }
      storages(tensorId) = tensor
      storages(storageId) = tensorStorage
    })
  }

  /**
   * Load weights from `modulePath` and copy to pre-defined module
   * for `layers` layers, copy all if not specified
   * @param definition  pre-defined module
   * @param modelPath   path where protobuf formatted module is stored
   * @param layers  name list of layers weight & bias of which to be copied
   * @param ev  numeric ops
   * @tparam T data type
   */

  def loadFromDefinition[T : ClassTag](definition : AbstractModule[Activity, Activity, T],
    modelPath : String, layers : mutable.HashSet[String] = null)(implicit ev: TensorNumeric[T])
  : Unit = {
    val loadedModule = loadFromFile(modelPath)
    val layersToCopy = if (layers == null) {
      val allLayers = new mutable.HashSet[String]()
      getAllLayers(definition, allLayers)
      allLayers
    } else {
      layers
    }
    copyParams(definition, loadedModule, layersToCopy)
  }

  private def getAllLayers[T : ClassTag](module : AbstractModule[Activity, Activity, T],
    layers : mutable.HashSet[String]) : Unit
    = {
    layers.add(module.getName)
    if (module.isInstanceOf[Container[_, _, _]]) {
      module.asInstanceOf[Container[_, _, _]].modules.foreach(subModule => {
        getAllLayers(subModule, layers)
      })
    }
  }

  private def copyParams[T : ClassTag](definition : AbstractModule[Activity, Activity, T],
                                     mirror : AbstractModule[Activity, Activity, T],
                                      layers : mutable.HashSet[String]) : Unit = {
    val parameterTable = definition.getParametersTable()
    val copiedParameterTable = mirror.getParametersTable()
    layers.foreach(name => {
      if (parameterTable.contains(name)) {
        require(copiedParameterTable.contains(name), s"$name does not exist in loaded module")
        copyParams(parameterTable.get(name).get.asInstanceOf[Table],
          copiedParameterTable.get(name).get.asInstanceOf[Table])
      }
    })
  }

  private def copyParams[T : ClassTag](params : Table, copyParams : Table) : Unit = {
    copyParam(params, copyParams, "weight")
    copyParam(params, copyParams, "bias")
  }

  private def copyParam[T : ClassTag](params : Table,
                                      copyParams : Table, paraName : String) : Unit = {
    if (params.contains(paraName)) {
      // this is for quantization tensors where the weight might be an array
      if (copyParams.get(paraName).get
        .isInstanceOf[Array[Tensor[T]]]) {
        require(params.get(paraName).get
          .isInstanceOf[Array[Tensor[T]]], "param type mismatch!")
        val copies = params.get(paraName).get
          .asInstanceOf[Array[Tensor[T]]]
        val origins = params.get(paraName).get
          .asInstanceOf[Array[Tensor[T]]]
        var i = 0
        while (i < copies.length) {
          origins(i).copy(copies(i))
          i += 1
        }
      } else {
        // For normal layers, their params are just tensors
        params.get(paraName).get.asInstanceOf[Tensor[T]].copy(
          copyParams.get(paraName).get.asInstanceOf[Tensor[T]])
      }
    }
  }
}

object ModulePersister {

  /**
   * Persist module to specified path
   * @param modelPath path to persist module to
   * @param module  module to be persisted
   * @param overwrite if overwrite module file if exists
   * @param ev  numeric ops
   * @tparam T data type
   */
  def saveToFile[T: ClassTag](modelPath: String,
                              weightPath: String = null,
                              module: AbstractModule[Activity, Activity, T],
                              overwrite: Boolean = false)
                             (implicit ev: TensorNumeric[T]): Unit = {

    if (weightPath == null) {
      val serializeResult = serializeModule(module, ProtoStorageType)
      setTensorStorage(serializeResult.bigDLModule, serializeResult.storages)
      File.saveBytes(serializeResult.bigDLModule.build.toByteArray, modelPath, overwrite)
    } else {
      val serializeResult = serializeModule(module, BigDLStorage)
      val tensorStorages = serializeResult.storages.filter(_._2.isInstanceOf[Array[_]])
      File.saveBytes(serializeResult.bigDLModule.build.toByteArray, modelPath, overwrite)
      saveWeightsToFile(weightPath, tensorStorages, overwrite)
    }
  }

  private def serializeModule[T : ClassTag](module: AbstractModule[Activity, Activity, T],
    storageType: StorageType)(implicit ev: TensorNumeric[T]): SerializeResult = {
    val bigDLModule = ModuleData(module
      , new ArrayBuffer[String](), new ArrayBuffer[String]())
    val storages = new mutable.HashMap[Int, Any]()
    val context = SerializeContext(bigDLModule, storages, storageType)
    ModuleSerializer.serialize(context)
  }

  private def saveWeightsToFile(weightPath: String, storages: mutable.HashMap[Int, Any],
    overwrite: Boolean = false): Unit = {
    val magicNo = SerConst.MAGIC_NO
    val total = storages.size
    var fw: FileWriter = null
    var out: OutputStream = null
    var objFile: ObjectOutputStream = null
    var digestOutputStream : DigestOutputStream = null
    var dataOutputStream: DataOutputStream = null
    try {
      fw = FileWriter(weightPath)
      out = fw.create(overwrite)
      val digest = MessageDigest.getInstance(SerConst.DIGEST_TYPE)
      digestOutputStream = new DigestOutputStream(out, digest);
      dataOutputStream = new DataOutputStream(digestOutputStream)
      digestOutputStream.on(true)
      dataOutputStream.writeInt(magicNo)
      dataOutputStream.writeInt(total)
      storages.foreach(storage => {
        val storageId = storage._1
        val dataArray = storage._2.asInstanceOf[Array[_]]
        val writer = DataReaderWriter(dataArray)
        dataOutputStream.writeInt(storageId)
        dataOutputStream.writeInt(writer.dataType().id)
        dataOutputStream.writeInt(dataArray.size)
        writer.write(dataOutputStream, dataArray)
      })
      digestOutputStream.on(false)
      val digestContent = digestOutputStream.getMessageDigest.digest
      dataOutputStream.writeInt(digestContent.length)
      dataOutputStream.write(digestContent)
    } finally {
      if (null != objFile) objFile.close()
      if (null != out) out.close()
      if (null != fw) fw.close()
      if (null != digestOutputStream) {
        digestOutputStream.flush()
        digestOutputStream.close()
      }
      if (null != dataOutputStream) {
        dataOutputStream.close()
      }
    }
  }


  private[bigdl] def setTensorStorage(bigDLModule: BigDLModule.Builder,
    storages: mutable.HashMap[Int, Any]) : Unit = {
    val storageIds = new mutable.HashSet[Int]
    val tensorStorages = storages.filter(_._2.isInstanceOf[TensorStorage])
    var nameAttributes = NameAttrList.newBuilder().setName(SerConst.GLOBAL_STORAGE)
    storages.values.filter(_.isInstanceOf[BigDLTensor]).foreach(storage => {
      val bigdlTensor = storage.asInstanceOf[BigDLTensor]
      val storageId = bigdlTensor.getStorage.getId
      if (!storageIds.contains(storageId) && storageId != -1) {
        val tensorBuilder = BigDLTensor.newBuilder(bigdlTensor)
        tensorBuilder.clearStorage()
        require(tensorStorages.contains(storageId), s"${storageId} does not exist")
        tensorBuilder.setStorage(tensorStorages.get(storageId).
          get.asInstanceOf[TensorStorage])
        val attrValueBuilder = AttrValue.newBuilder
        attrValueBuilder.setTensorValue(tensorBuilder.build)
        nameAttributes.putAttr(tensorBuilder.getId.toString, attrValueBuilder.build)
        storageIds.add(storageId)
      }
    })
    val attrValueBuilder = AttrValue.newBuilder
    attrValueBuilder.setNameAttrListValue(nameAttributes)
    bigDLModule.putAttr(SerConst.GLOBAL_STORAGE, attrValueBuilder.build)
  }

  /**
   * Save module definition to given path
   * @param definitionPath the path to persist definition path to
   * @param module  module to be persisted
   * @param overwrite if overwrite module file if exists
   * @param ev numeric ops
   * @tparam T data type
   */
  def saveModelDefinitionToFile[T: ClassTag](definitionPath : String,
    module : AbstractModule[Activity, Activity, T],
    overwrite : Boolean = false)(implicit ev: TensorNumeric[T]) : Unit = {
    val bigDLModule = ModuleData(module, new ArrayBuffer[String](), new ArrayBuffer[String]())
    val storages = new mutable.HashMap[Int, Any]()
    val context = SerializeContext(bigDLModule, storages, ProtoStorageType)
    val bigDLModel = ModuleSerializer.serialize(context).bigDLModule
    cleantWeightAndBias(bigDLModel)
    val model = bigDLModel.build
    val byteArrayOut = new ByteArrayOutputStream()
    byteArrayOut.write(model.toString.getBytes)
    File.saveBytes(byteArrayOut.toByteArray, definitionPath, overwrite)
  }

  private def cleantWeightAndBias(modelBuilder : BigDLModule.Builder): Unit = {
    modelBuilder.clearWeight
    modelBuilder.clearBias
    if (modelBuilder.getSubModulesCount > 0) {
      val subModules = modelBuilder.getSubModulesList
      modelBuilder.clearSubModules
      subModules.asScala.foreach(sub => {
        val subModelBuilder = BigDLModule.newBuilder(sub)
        cleantWeightAndBias(subModelBuilder)
        modelBuilder.addSubModules(subModelBuilder.build)
      })
    }
  }
}
