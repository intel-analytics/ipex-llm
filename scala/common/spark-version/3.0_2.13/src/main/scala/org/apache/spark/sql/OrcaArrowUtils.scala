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

import java.io.{DataOutputStream, FileInputStream, FileOutputStream}

import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.arrow.vector.ipc.ArrowStreamWriter
import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.api.python.PythonSQLUtils
import org.apache.spark.sql.catalyst.{CatalystTypeConverters, InternalRow}
import org.apache.spark.sql.execution.arrow.{ArrowConverters, ArrowWriter}
import org.apache.spark.sql.execution.python.BatchIterator
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.util.Utils

import org.apache.spark.util.{ShutdownHookManager, Utils}

import java.io._


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
}
