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
import java.util.Base64

import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}
import org.apache.arrow.vector.{FieldVector, Float4Vector, IntVector, VectorSchemaRoot}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.serving.serialization.ArrowSerializer.getSchema
import scala.collection.JavaConverters._

class ArrowSerializerSpec extends FlatSpec with Matchers {
  "Arrow Serialization raw data" should "work" in {
    val allocator = new RootAllocator(Int.MaxValue)
    val data = Array(1.2f, 2, 4, 5, 6)
    val shape = Array(1, 2, 3)
    val ser = new ArrowSerializer(data, shape)
    val vectorSchemaRoot = VectorSchemaRoot.create(getSchema, allocator)
    vectorSchemaRoot.setRowCount(5)
    ser.copyToSchemaRoot(vectorSchemaRoot)
    val out = new ByteArrayOutputStream()
    val writer = new ArrowStreamWriter(vectorSchemaRoot, null, out)
    writer.start()
    writer.writeBatch()
    writer.end()
    out.flush()
    out.close()
    val byteArr = out.toByteArray
    val str = Base64.getEncoder.encodeToString(byteArr)
    val readAllocator = new RootAllocator(Int.MaxValue)
    val reader = new ArrowStreamReader(new ByteArrayInputStream(byteArr), readAllocator)
    val schema = reader.getVectorSchemaRoot.getSchema
    while (reader.loadNextBatch()) {
      val schemaRoot = reader.getVectorSchemaRoot
      schemaRoot
    }
    require(schema.getFields.size() == 2, "schema number wrong.")
  }
  "Arrow example" should "work" in {
    val allocator = new RootAllocator(Int.MaxValue)
    val dataVector = new Float4Vector("boolean", allocator)
    val shapeVector = new IntVector("varchar", allocator)
    val data = Array(1.2f, 2, 4, 5, 6)
    val shape = Array(1, 2, 3)
    val ser = new ArrowSerializer(data, shape)
    ser.copyDataToVector(dataVector)
    ser.copyShapeToVector(shapeVector)
    val fields = List(dataVector.getField, shapeVector.getField)
    val vectors = List[FieldVector](dataVector, shapeVector)
    val vectorSchemaRoot = new VectorSchemaRoot(fields.asJava, vectors.asJava)

    val out = new ByteArrayOutputStream()
    val writer = new ArrowStreamWriter(vectorSchemaRoot, null, out)
    writer.start()
    writer.writeBatch()
    writer.end()
    out.flush()
    out.close()
    val readAllocator = new RootAllocator(Int.MaxValue)
    val reader = new ArrowStreamReader(new ByteArrayInputStream(out.toByteArray), readAllocator)
    val schema = reader.getVectorSchemaRoot.getSchema
    while (reader.loadNextBatch()) {
      val schemaRoot = reader.getVectorSchemaRoot
      schemaRoot.getFieldVectors.asScala.foreach(v => {
        require(v.getValueCount == 5, "vector size wrong")
      })
    }
    require(schema.getFields.size() == 2, "schema number wrong.")
  }

}
