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

import com.intel.analytics.bigdl.nn.Identity
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.quantized.Utils.ModuleNode
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.bytedeco.onnx.{GraphProto, NodeProto, TensorProto, TypeProto_Tensor, ValueInfoProto}
import org.bytedeco.onnx.global.onnx.AttributeProto_AttributeType_INTS
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.collection.immutable.HashMap
import scala.collection.mutable
import scala.reflect.ClassTag
import scala.tools.scalap.ByteArrayReader



case class TensorValueInfo(shape: Array[Int], elemType: Int)


class BigDLGraph[T: ClassTag] (implicit ev: TensorNumeric[T]) {

  // The name of the graph
  var name: String = null
  var docString: String = null

  var initializer: List[Int] = null

  var input: List[Int] = null

  var output: List[Int] = null

  var valueInfo: List[Int] = null

  var quantizationAnnotation: List[Int] = null

  val module: AbstractModule[Activity, Activity, Float] = null


  def getName(): String = name
  def setName(newName: String): Unit = {
    this.name = newName
  }

  def getDocString(): String = docString
  def setDocString(newDocString: String): Unit = {

  }

  def getInitializer(): List[Int] = initializer
  def setInitializer(newInitList: List[Int]): Unit = {
    initializer = newInitList
  }

  def getInput(): List[Int] = input
  def setInput(inputList: List[Int]): Unit = {
    input = inputList
  }

  def getOutput(): List[Int] = output
  def setOutput(outputList: List[Int]): Unit = {
    output = outputList
  }

  def getValueInfo(): List[Int] = valueInfo
  def setValueInfo(valueInfoList: List[Int]): Unit = {
    valueInfo = valueInfoList
  }

  def getQuantizationAnnotation(): List[Int] = quantizationAnnotation
  def setQuantizationAnnotation(annotation: List[Int]): Unit = {
    quantizationAnnotation = annotation
  }


  private def initTensorLookup(graph: GraphProto): mutable.Map[String, Array[Int]] = {

    val initTensor = (0 until graph.initializer_size()).map(i => {
      val currInitTensor = graph.initializer(i)
      val currTensorDim = (0 until currInitTensor.dims_size()).map(i => {
        currInitTensor.dims(i).toInt
      }).toArray[Int]
      val currTensorData = currInitTensor.raw_data().getStringBytes

    })

    val inputTensorLookup = (0 until graph.input_size()).map(i => {
      val currInput = graph.input(i)
      require(currInput.has_type())
      require(currInput.`type`().has_tensor_type())
      require(currInput.has_name())

      val currName = currInput.name().getString

      val currInputTensor = currInput.`type`().tensor_type()

      val currTensorType = currInputTensor.elem_type()
      val currTensorDims = (0 until currInputTensor.shape().dim_size()).map(j => {
        require(currInputTensor.shape().dim(j).has_dim_value())
        currInputTensor.shape().dim(j).dim_value().toInt
      }).toArray



      (currName, currTensorDims)
    }
    ).toMap

    val outputTensorLookup = (0 until graph.output_size()).map(i => {
      val currOutput = graph.output(i)
      require(currOutput.has_type())
      require(currOutput.`type`().has_tensor_type())
      require(currOutput.has_name())

      val currName = currOutput.name().getString
      val currInputTensor = currOutput.`type`().tensor_type()
      println("output", i, currName)

      val currTensorType = currInputTensor.elem_type()
      val currTensorDims = (0 until currInputTensor.shape().dim_size()).map(j => {
        require(currInputTensor.shape().dim(j).has_dim_value())
        currInputTensor.shape().dim(j).dim_value().toInt
      }).toArray

      (currName, currTensorDims)
    }
    ).toMap

    mutable.Map((inputTensorLookup ++ outputTensorLookup).toSeq: _*)
  }


  private def getInitTensorMap(graph: GraphProto): mutable.Map[String, (Int,
    Array[Int], Array[Any])] = {

    val initTensorMap = new mutable.HashMap[String, (Int, Array[Int], Array[Any])]()

    (0 until graph.initializer_size()).map(i => {

      val currTensor = graph.initializer(i)

//      println(currTensor.name().getString)

      require(currTensor.has_name())
      require(currTensor.has_data_type())
//      require(currTensor.has_data_location())
      require(currTensor.has_segment() == false)

      val currTensorName = currTensor.name().getString
      val dataType = currTensor.data_type()

      val currTensorDims = (0 until currTensor.dims_size()).map(i =>
        currTensor.dims(i).toInt
      ).toArray[Int]

      if (currTensor.has_raw_data()) {
        if (dataType == TensorProto.UNDEFINED || dataType == TensorProto.STRING) {
          throw new IllegalArgumentException
        }
        val rawData = currTensor.raw_data().getStringBytes
        val baReader: ByteArrayReader = new ByteArrayReader(rawData)

        val dataArray = dataType match {
          case TensorProto.FLOAT =>
            (0 until rawData.length / 4).map(i => {
              baReader.getFloat(4 * i)
            }).toArray[Any]
          case TensorProto.DOUBLE =>
            (0 until rawData.length / 4).map(i => {
              baReader.getDouble(4 * i)
            }).toArray[Any]
          case TensorProto.INT64 =>
            (0 until rawData.length / 8).map(i => baReader.getLong(8 * i)).toArray[Any]
          case _ => throw new UnsupportedOperationException()
        }

        initTensorMap.put(currTensorName, (dataType, currTensorDims, dataArray))

      } else if (currTensor.float_data_size() > 0) {
        // getFloatArray(currTensor, currTensor.float_data_size())
        null
      } else if (currTensor.int32_data_size() > 0) {
        // getInt32Array(currTensor, currTensor.int32_data_size())
        null
      } else if (currTensor.string_data_size() > 0) {
        null
      } else if (currTensor.int64_data_size() > 0) {
        // getInt64Array(currTensor, currTensor.int64_data_size())
      } else {
        null
      }
    })

    initTensorMap

  }


  private def makeOutputTensor(node: NodeProto): TensorProto = {
    val outTensor = new TensorProto()

    node.input(0)
    node.input(1)
    outTensor
  }

  def makeNode(): Unit = {
    val module = new Identity[Double]()

  }
}


object BigDLGraph {

  def fromOnnx[T: ClassTag](graph: GraphProto)(implicit ev: TensorNumeric[T]): BigDLGraph[Float] = {
    val bGraph = new BigDLGraph[Float]()

    val graphName = if (graph.has_name()) graph.name().getString else null

    val docString = if (graph.has_doc_string()) graph.doc_string() else null


    val nodeLookup: mutable.HashMap[String, NodeProto] = new mutable.HashMap[String, NodeProto]()
    val tensorLookup = bGraph.initTensorLookup(graph)


    (0 until graph.node_size()).foreach(i => {
      val currNode = graph.node(i)
      val opType = currNode.op_type().getString
      println("op type", opType)
      val bModule = OperationConverter.convertOp(currNode, tensorLookup)
//      val nodeOutput = bModule.output
      val nodeOutput = Array[Int](2, 2, 3, 224)

      if (currNode.output_size() == 1) {
        tensorLookup.put(currNode.output(0).getString(), nodeOutput)
      } else {
        val outTable = nodeOutput.asInstanceOf[Table]
        (0 until currNode.output_size()).foreach(i => {
          val outTensorName = currNode.output(i).getString
          val outTensorDims = currNode.output(i).asInstanceOf[Tensor[Float]].size()
          tensorLookup.put(outTensorName, outTensorDims)
        })
      }


    })


    nodeLookup.foreach(node => {
      val nodeName = node._1
      val nodeProto = node._2
//      println(nodeName, nodeProto.op_type().getString)
      val opType = nodeProto.op_type().getString
      if (opType == "Conv") {
        println(nodeProto.input(0).getString, nodeProto.input(1).getString, nodeName)
        val test: ModuleNode[Float] = OperationConverter.convertOp(nodeProto, tensorLookup)

      }
    })

    graph.value_info_size()

    println("quantization size", graph.quantization_annotation_size())

    val quantizationAnnos = mutable.ListBuffer[TensorAnnotation]()
    for (i <- 0 until graph.quantization_annotation_size()) {
      val currAnno = graph.quantization_annotation(i)
      val currAnnoTensorName = currAnno.tensor_name().getString
      val currAnnoParamSize = currAnno.quant_parameter_tensor_names_size()
      (0 until currAnnoParamSize).map(i => {
        (currAnno.quant_parameter_tensor_names(i).key().getString(),
          currAnno.quant_parameter_tensor_names(i).value().getString)
      }).toMap
    }

    bGraph
  }

  def toOnnx[T: ClassTag](bigdlGraph: BigDLGraph[T]) (implicit ev: TensorNumeric[T]): GraphProto = {
    val gProto = new GraphProto()

    gProto
  }

}
