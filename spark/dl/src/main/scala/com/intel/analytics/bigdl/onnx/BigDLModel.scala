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

package com.intel.analytics.bigdl.onnx



import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import org.bytedeco.javacpp._
import org.bytedeco.onnx._
import org.bytedeco.onnx.global.onnx._

import scala.collection.mutable


class BigDLModel {

  // The version of the IR this model targets. See Version enum above.
  // This field MUST be present.
  var irVersion: Long = 0

  // The OperatorSets this model relies on.
  // All ModelProtos MUST have at least one entry that
  // specifies which version of the ONNX OperatorSet is
  // being imported.
  //
  // All nodes in the ModelProto's graph will bind against the operator
  // with the same-domain/same-op_type operator with the HIGHEST version
  // in the referenced operator sets.
  // repeated OperatorSetIdProto opset_import = 8;
  val opsetImport: mutable.HashMap[String, Long] = mutable.HashMap()

  // The name of the framework or tool used to generate this model.
  // This field SHOULD be present to indicate which implementation/tool/framework
  // emitted the model.
  // optional string producer_name = 2;
  var producerName: String = ""

  // The version of the framework or tool used to generate this model.
  // This field SHOULD be present to indicate which implementation/tool/framework
  // emitted the model.
  // optional string producer_version = 3;
  var producerVersion: String = ""

  // Domain name of the model.
  // We use reverse domain names as name space indicators. For example:
  // `com.facebook.fair` or `com.microsoft.cognitiveservices`
  //
  // Together with `model_version` and GraphProto.name, this forms the unique identity of
  // the graph.
  // optional string domain = 4;
  var domain: String = ""

  // The version of the graph encoded. See Version enum below.
  // optional int64 model_version = 5;
  var modelVersion: Long = 0

  // A human-readable documentation for this model. Markdown is allowed.
  // optional string doc_string = 6;
  var docString: String = ""

  // The parameterized graph that is evaluated to execute the model.
  // optional GraphProto graph = 7;
  val graph: BigDLGraph = null

  // Named metadata values; keys should be distinct.
  // repeated StringStringEntryProto metadata_props = 14;
  val metadataProps: List[Int] = List()



  def getIRVersion(): Long = {
    irVersion
  }

  def setIRVersion(version: Long): Unit = {

  }

  def getOptsetImport(): mutable.HashMap[String, Long] = {
    opsetImport
  }

  def setOpsetImport(): Unit = {

  }

  def getProducerName(): Unit = {

  }

  def setProducerName(name: String): Unit = {
    producerName = name
  }

  def getProducerVersion(): String = {
    producerVersion
  }

  def setProducerVersion(version: String): Unit = {
    producerVersion = version
  }

  def getDomain(): String = {
    domain
  }

  def setDomain(domain: String): Unit = {
    this.domain = domain
  }

  def getModelVersion(): Long = {
    modelVersion
  }

  def setModelVersion(version: Long): Unit = {
    modelVersion = version
  }

  def getDocString(): String = {
    docString
  }

  def setDocString(doc: String): Unit = {
    docString = doc
  }

  override def toString: String = super.toString
}


object BigDLModel {

  def load(fileName: String): BigDLModel = {
    val allSchemas: OpSchemaVector = OpSchemaRegistry.get_all_schemas();
    println(allSchemas.size())

    val onnxModel: ModelProto = new ModelProto()

    val modelBytes: Array[Byte] = Files.readAllBytes(Paths.get(fileName))
    val modelByteBuffer: ByteBuffer = ByteBuffer.wrap(modelBytes);

    ParseProtoFromBytes(onnxModel, new BytePointer(modelByteBuffer), modelBytes.length)

    check_model(onnxModel)

    // InferShapes(onnxModel)

    val passes: StringVector = new StringVector("eliminate_nop_transpose", "eliminate_nop_pad",
      "fuse_consecutive_transposes", "fuse_transpose_into_gemm");

    Optimize(onnxModel, passes)

    this.convertModel(onnxModel)

//    throw new UnsupportedOperationException("Unimplemented")

  }

  private def convertModel(model: ModelProto): BigDLModel = {
    val convertedModel = new BigDLModel()

    if (model.has_ir_version()) {
      convertedModel.setIRVersion(model.ir_version())
    } else {

    }

    if (model.has_producer_name()) {
      convertedModel.setProducerName(model.producer_name().getString())
    } else {

    }

    if (model.has_producer_version()) {
      convertedModel.setProducerVersion(model.producer_version().getString())
    } else {

    }

    if (model.has_domain()) {
      convertedModel.setDomain(model.domain().getString())
    } else {

    }

    if (model.has_model_version()) {
      convertedModel.setModelVersion(model.model_version())
    } else {

    }

    if (model.has_doc_string()) {
      convertedModel.setDocString(model.doc_string().getString())
    } else {

    }


    val opsetSize = model.opset_import_size()
    for(i <- 0 until opsetSize) {
      val currOp = model.opset_import(i)
      val opDomain = if (currOp.has_domain()) currOp.domain().getString else null
      val opVersion =
        if (currOp.has_version()) currOp.version() else throw new IllegalArgumentException()
      convertedModel.getOptsetImport().put(opDomain, opVersion)
    }
    val bigdlGraph = convertGraph(model.graph())

    convertedModel
  }

  private def convertGraph(proto: GraphProto): BigDLGraph = {
    null
  }

  def save(fileName: String): Unit = {
    throw new UnsupportedOperationException("Unimplemented")
  }

  val modelPath = "/home/leicongl/Workground/myData/models/onnx/alexnet.onnx"

  def main(args: Array[String]): Unit = {

    val allSchemas: OpSchemaVector = OpSchemaRegistry.get_all_schemas();
    println(allSchemas.size())

    load(modelPath)

  }


  def modelSummary(model: ModelProto): Unit = {

    println("IR Version", model.ir_version())
    println("Producer", model.producer_name().getString, model.producer_version().getString)
    println("Domain", model.domain())
    println("Model version", model.model_version())
    println("Model DocString", model.doc_string())

    val irVersion = model.ir_version()
    val producerName = model.producer_name().getString
    val producerVersion = model.producer_version().getString
    val domain = model.domain().getString
    val modelVersion = model.model_version()
    val docString = model.doc_string().getString

    println("OpSet", model.opset_import_size(),
      model.opset_import(0).domain(), model.opset_import(0).version())

    // graphSummary(model.graph())


  }

  def graphSummary(graph: GraphProto): Unit = {
    println("graph name", graph.name().getString)
    println("node size", graph.node_size())
    println("input size", graph.input_size())
    println("output size", graph.output_size())
    println("initializer", graph.initializer_size())

    //    val tempFloat: Float = 3.1415926.toFloat
    //    val tempByteArray = floatToByteArray(tempFloat)
    //    val tempByteReader = new ByteArrayReader(tempByteArray)
    //
    //    println("TEMP BYTE ARRAY", tempByteReader.nextFloat)

    for(i <- 0 until graph.initializer_size()) {
      val currInit = graph.initializer(i)
      println("initializer#", i, currInit.name().getString)
      println("dim size", currInit.dims_size())
      println("float data size", currInit.float_data_size())
      println("data location", currInit.data_location())
      println("raw data", currInit.has_raw_data(), currInit.raw_data().getStringBytes.length)

      // val baReader: ByteArrayReader = new ByteArrayReader(currInit.raw_data().getStringBytes)
      //      (0 until currInit.raw_data().getStringBytes.length / 4).foreach(i =>
      //        println(baReader.nextFloat, baReader.getFloat(i * 4))
      //      )


    }


    println("value info", graph.value_info_size())

    println("quantization anno", graph.quantization_annotation_size())

    (0 until graph.node_size()).foreach(i => nodeSummary(graph.node(i)))

    println("value info", graph.value_info_size())
    (0 until graph.value_info_size()).foreach(i =>
      println(graph.value_info(i).name().getString)

    )
  }


  def nodeSummary(node: NodeProto): Unit = {
    println("name", if (node.has_name()) node.name().getString else None)
    println("doc string", if (node.has_doc_string()) node.doc_string().getString else None)
    println("domain", if (node.has_domain()) node.domain().getString else None)
    println("op type", if (node.has_op_type()) node.op_type().getString else None)
    println("input size", node.input_size())
    (0 until node.input_size()).foreach(i => println(node.input(i).getString))
    println("output size", node.output_size())
    (0 until node.output_size()).foreach(i => println(node.output(i).getString))
    println("attributes size", node.attribute_size())
    (0 until node.attribute_size()).foreach(i => attributeSummary(node.attribute(i)))

  }

  def attributeSummary(attr: AttributeProto): Unit = {
    println("attribute name", attr.name().getString)
    println("attribute type", attr.`type`())
    println("attribute value", (0 until attr.ints_size()).map(i => attr.ints(i)))
  }

  def floatToByteArray(value: Float): Array[Byte] = {
    val intBits: Int = java.lang.Float.floatToIntBits(value)
    return Array (
      (intBits >> 24).toByte, (intBits >> 16).toByte, (intBits >> 8).toByte, (intBits).toByte)

  }

}
