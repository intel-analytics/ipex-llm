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

package org.apache.spark.sql

import org.apache.arrow.memory.{AllocationListener, RootAllocator}

import java.io.{DataOutputStream, FileInputStream, FileOutputStream}
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}
import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.api.python.PythonSQLUtils
import org.apache.spark.sql.catalyst.{CatalystTypeConverters, InternalRow}
import org.apache.spark.sql.execution.arrow.{ArrowConverters, ArrowWriter}
import org.apache.spark.sql.execution.python.BatchIterator
import org.apache.spark.sql.types.{ArrayType, DataType, FloatType, StructField, StructType}
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.util.Utils
import org.apache.spark.util.{ShutdownHookManager, Utils}

import java.io._
import javax.xml.bind.DatatypeConverter
import scala.collection.mutable.ArrayBuffer
import java.util.{List => JList}
import scala.collection.JavaConverters._


class OrcaArrowUtils() {
  def orcaToDataFrame(jrdd: JavaRDD[String], schemaString: String,
  sqlContext: SQLContext): DataFrame = {
    val schema = DataType.fromJson(schemaString).asInstanceOf[StructType]
    val timeZoneId = sqlContext.sessionState.conf.sessionLocalTimeZone
    val rdd = jrdd.rdd.mapPartitions { iter =>
      val context = TaskContext.get()
      val file = iter.next()
      val dir = new File(file)
      ShutdownHookManager.registerShutdownDeleteDir(dir)

      Utils.tryWithResource(new FileInputStream(file)) { fileStream =>
        // Create array to consume iterator so that we can safely close the file
        val batches = ArrowConverters.getBatchesFromStream(fileStream.getChannel)
        ArrowConverters.fromBatchIterator(batches,
          DataType.fromJson(schemaString).asInstanceOf[StructType], timeZoneId, context)
      }
    }
    sqlContext.internalCreateDataFrame(rdd.setName("arrow"), schema)
  }

  // Below code is adapted from spark, https://github.com/apache/spark/blob/branch-3.1/sql/core/
  // src/main/scala/org/apache/spark/sql/execution/python/ArrowPythonRunner.scala
  def sparkdfTopdf(sdf: DataFrame, sqlContext: SQLContext, batchSize: Int = -1): RDD[String] = {
    val schemaCaptured = sdf.schema
    val maxRecordsPerBatch = if (batchSize == -1) {
      sqlContext.sessionState.conf.arrowMaxRecordsPerBatch
    } else batchSize
    val timeZoneId = sqlContext.sessionState.conf.sessionLocalTimeZone


    val schema = sdf.schema
    sdf.rdd.mapPartitions {iter =>
      val batchIter = if (maxRecordsPerBatch > 0) {
        new BatchIterator(iter, maxRecordsPerBatch)
      } else Iterator(iter)

      val arrowSchema = ArrowUtils.toArrowSchema(schema, timeZoneId)
      val allocator = ArrowUtils.rootAllocator.newChildAllocator("ItertoFile", 0, Long.MaxValue)
      val root = VectorSchemaRoot.create(arrowSchema, allocator)

      val conf = SparkEnv.get.conf
      val sparkFilesDir =
        Utils.createTempDir(Utils.getLocalDir(conf), "arrowCommunicate").getAbsolutePath

      val filename = sparkFilesDir + "/arrowdata"
      val fos = new FileOutputStream(filename)
      val dataOutput = new DataOutputStream(fos)

      Utils.tryWithSafeFinally {
        val arrowWriter = ArrowWriter.create(root)
        val writer = new ArrowStreamWriter(root, null, dataOutput)
        writer.start()

        while (batchIter.hasNext) {
          val nextBatch = batchIter.next()

          while (nextBatch.hasNext) {
            val nxtIternalRow = CatalystTypeConverters.convertToCatalyst(nextBatch.next())
            arrowWriter.write(nxtIternalRow.asInstanceOf[InternalRow])
          }

          arrowWriter.finish()
          writer.writeBatch()
          arrowWriter.reset()
        }
        writer.end()
      } {
        root.close()
        allocator.close()
        if (dataOutput != null) {
          dataOutput.close()
        }
        if (fos != null) {
          fos.close()
        }
      }
      Iterator(filename)
    }
  }

  def openVINOOutputToSDF(df: DataFrame,
                          outputRDD: JavaRDD[String],
                          outputNames: JList[String],
                          outShapes: JList[JList[Int]]): DataFrame = {
    val spark = SparkSession.builder.config(outputRDD.sparkContext.getConf).getOrCreate()
    val outputNamesScala = outputNames.asScala
    val outputShapesScala = outShapes.asScala.map(_.asScala.toArray[Int]).toArray
    val outputShapeReverseArr = outputShapesScala.map(_.reverse.dropRight(1))
    val outputRowRDD = outputRDD.rdd.flatMap(hexStr => {
      val allocator = new RootAllocator()
      val in = new ByteArrayInputStream(DatatypeConverter.parseHexBinary(hexStr))
      val stream = new ArrowStreamReader(in, allocator)
      val vsr = stream.getVectorSchemaRoot
      val outputVectorReaders = outputNamesScala.map(name => {
        vsr.getVector(name).getReader
      })
      // Only one batch in stream
      stream.loadNextBatch()
      val rowCount = vsr.getRowCount
      val outputRowSeq = (0 until rowCount).map(i => {
        val row_vector = (outputVectorReaders zip outputShapeReverseArr).map(readerShapeTuple => {
          val reader = readerShapeTuple._1
          reader.setPosition(i)
          val shape = readerShapeTuple._2
          val dataArr = ArrayBuffer[Float]()
          while (reader.next()) {
            val floatReader = reader.reader()
            if (floatReader.isSet) {
              dataArr += floatReader.readFloat()
            }
          }
          // OpenVINO output dim >= 1
          if (shape.length == 0) {
            dataArr.toArray
          } else {
            // Reshape if dim >= 2
            var groupedArr: Array[Any] = dataArr.toArray
            for (s <- shape) {
              groupedArr = groupedArr.grouped(s).toArray
            }
            groupedArr
          }
        })
        Row.fromSeq(row_vector)
      })
      stream.close()
      in.close()
      allocator.close()
      outputRowSeq
    })

    val mergedRDD = df.rdd.zip(outputRowRDD).map(originOutputRowTuple => {
      Row.fromSeq(originOutputRowTuple._1.toSeq ++ originOutputRowTuple._2.toSeq)
    })

    val originSchema = df.schema
    val resultStruct = (outputNamesScala zip outputShapesScala).map(nameShape => {
      var structType: DataType = FloatType
      for (_ <- nameShape._2.indices) {
        structType = ArrayType(structType)
      }
      StructField(nameShape._1, structType, true)
    }).toArray
    val schema = StructType(originSchema.fields ++: resultStruct)
    spark.createDataFrame(mergedRDD, schema)
  }
}
