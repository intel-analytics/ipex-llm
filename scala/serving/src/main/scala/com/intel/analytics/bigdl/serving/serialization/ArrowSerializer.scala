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

package com.intel.analytics.bigdl.serving.serialization

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.serving.utils.{Conventions, TensorUtils}
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.complex.{BaseRepeatedValueVector, ListVector}
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.apache.arrow.vector.{FieldVector, Float4Vector, IntVector, VectorSchemaRoot}

import scala.collection.JavaConverters._

/**
 * An ArrowSerializer is an instance to handle Arrow serialization of a Tensor
 * Mulitple Tensor output, aka, Table, is processed by object ArrowSerializaer
 * which construct multiple ArrowSerializer
 * @param data data array of flattened Tensor
 * @param shape shape array of Tensor
 */
class ArrowSerializer(data: Array[Float], shape: Array[Int]) {
  /**
   * Copy data array to Arrow float vector
   * @param vector
   */
  def copyDataToVector(vector: Float4Vector): Unit = {
    vector.allocateNew(data.size)
    (0 until data.size).foreach(i => vector.set(i, data(i)))
    vector.setValueCount(data.size)
  }
  /**
   * Copy shape array to Arrow shape vector
   * @param vector
   */
  def copyShapeToVector(vector: IntVector): Unit = {
    vector.allocateNew(shape.size)
    (0 until shape.size).foreach(i => vector.set(i, shape(i)))
    vector.setValueCount(data.size)
  }
  /**
   * Copy the data and shape in this class into vectorSchemaRoot
   * vectorSchemaRoot is then used to store the data
   * which could be written to Arrow buffer
   * @param vectorSchemaRoot
   */
  def copyToSchemaRoot(vectorSchemaRoot: VectorSchemaRoot): Unit = {
    vectorSchemaRoot.setRowCount(data.size)
    val dataVector = vectorSchemaRoot.getVector("data").asInstanceOf[Float4Vector]
    val shapeVector = vectorSchemaRoot.getVector("shape").asInstanceOf[IntVector]
    copyDataToVector(dataVector)
    copyShapeToVector(shapeVector)
  }
  /**
   * deprecated
   * Create a vector of data and shape of type ListVector
   * @return
   */
  def createVector(): VectorSchemaRoot = {
    val vectorSchemaRoot = VectorSchemaRoot.create(
      ArrowSerializer.getSchema, new RootAllocator(Integer.MAX_VALUE))
    // copy data into data ListVector
    val dataList = vectorSchemaRoot.getVector("data").asInstanceOf[ListVector]
    val dataVector = dataList.getDataVector.asInstanceOf[Float4Vector]
    dataList.startNewValue(0)
    copyDataToVector(dataVector)
    dataList.endValue(0, data.size)
    dataList.setValueCount(1)
    // copy shape into shape ListVector
    val shapeList = vectorSchemaRoot.getVector("shape").asInstanceOf[ListVector]
    val shapeVector = shapeList.getDataVector.asInstanceOf[IntVector]
    shapeList.startNewValue(0)
    copyShapeToVector(shapeVector)
    shapeList.endValue(0, shape.size)
    shapeList.setValueCount(1)

    vectorSchemaRoot
  }
}

/**
 * Object ArrowSerializer wraps the operations in class ArrowSerializer
 * and could get either Tensor or Table and serialize it to array of byte
 */
object ArrowSerializer {
  /**
   * get Arrow schema of Tensor, which is regular schema of data and shape
   * @return
   */
  def getSchema: Schema = {
    val dataField = new Field("data", FieldType.nullable(Conventions.ARROW_FLOAT), null)
    val shapeField = new Field("shape", FieldType.nullable(Conventions.ARROW_INT), null)
    new Schema(List(dataField, shapeField).asJava, null)
  }
  /**
   * Write Tensor to VectorSchemaRoot, and use ArrowStreamWriter to write
   * The ArrowStreamWriter is pointed to ByteArrayOutputStream in advance
   * @param tensor Tensor to write
   * @param vectorSchemaRoot VectorSchemaRoot to store data of Tensor
   * @param writer ArrowStreamWriter to write Tensor
   */
  def writeTensor(tensor: Tensor[Float],
                  vectorSchemaRoot: VectorSchemaRoot,
                  writer: ArrowStreamWriter): Unit = {
    val shape = tensor.size()
    val totalSize = TensorUtils.getTotalSize(tensor)
    val data = tensor.resize(totalSize).toArray()
    val serializer = new ArrowSerializer(data, shape)
    serializer.copyToSchemaRoot(vectorSchemaRoot)
    writer.writeBatch()
  }

  /**
   * Convert Activity batch to array of byte
   * @param t Activity to convert
   * @param idx index of Activity to convert in this batch
   * @return array of byte converted
   */
  def activityBatchToByte(t: Activity, idx: Int): Array[Byte] = {
    val allocator = new RootAllocator(Int.MaxValue)
    val out = new ByteArrayOutputStream()
    val vectorSchemaRoot = VectorSchemaRoot.create(getSchema, allocator)
    val writer = new ArrowStreamWriter(vectorSchemaRoot, null, out)
    writer.start()
    if (t.isTable) {
      t.toTable.keySet.foreach(key => {
        val tensor = if (idx > 0) {
          t.toTable(key).asInstanceOf[Tensor[Float]].select(1, idx)
        } else {
          t.toTable(key).asInstanceOf[Tensor[Float]]
        }
        writeTensor(tensor, vectorSchemaRoot, writer)
      })

    } else if (t.isTensor) {
      val tensor = if (idx > 0) {
        t.toTensor[Float].select(1, idx)
      } else {
        t.toTensor[Float]
      }
      writeTensor(tensor, vectorSchemaRoot, writer)
    } else {
      throw new Error("Your input for Post-processing is invalid, " +
        "neither Table nor Tensor, please check.")
    }
    vectorSchemaRoot.close()
    writer.end()
    writer.close()
    out.flush()
    out.close()
    out.toByteArray
  }
}
