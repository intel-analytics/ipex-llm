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

import java.io.FileInputStream

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.sql.api.python.PythonSQLUtils
import org.apache.spark.sql.execution.arrow.ArrowConverters

import scala.reflect.ClassTag

object PythonOrcaSQLUtils {

  def ofFloat(): PythonOrcaSQLUtils[Float] = new PythonOrcaSQLUtils[Float]()

  def ofDouble(): PythonOrcaSQLUtils[Double] = new PythonOrcaSQLUtils[Double]()
}

class PythonOrcaSQLUtils[T: ClassTag](implicit ev: TensorNumeric[T]) {
  def readArrowStreamFromFile(file: String): Array[Array[Byte]] = {
    org.apache.spark.util.Utils.tryWithResource(new FileInputStream(file)) { fileStream =>
      // Create array to consume iterator so that we can safely close the file
      ArrowConverters.getBatchesFromStream(fileStream.getChannel).toArray
    }
  }

  def orcaToDataFrame(arrowBatchRDD: JavaRDD[Array[Byte]],
                  schemaString: String,
                  sqlContext: SQLContext): DataFrame = {
    PythonSQLUtils.toDataFrame(arrowBatchRDD, schemaString, sqlContext)
  }
}
