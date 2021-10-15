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

package com.intel.analytics.bigdl.orca.python

import com.intel.analytics.bigdl.orca.inference.InferenceModel
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import java.util.{List => JList}

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.common.PythonZoo

import scala.reflect.ClassTag

object PythonOrca {

  def ofFloat(): PythonOrca[Float] = new PythonOrca[Float]()

  def ofDouble(): PythonOrca[Double] = new PythonOrca[Double]()
}

class PythonOrca[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  def inferenceModelDistriPredict(model: InferenceModel, sc: JavaSparkContext,
                                  inputs: JavaRDD[JList[com.intel.analytics.bigdl.dllib.
                                  utils.python.api.JTensor]],
                                  inputIsTable: Boolean): JavaRDD[JList[Object]] = {
    val broadcastModel = sc.broadcast(model)
    inputs.rdd.mapPartitions(partition => {
      val localModel = broadcastModel.value
      partition.map(inputs => {
        val inputActivity = jTensorsToActivity(inputs, inputIsTable)
        val outputActivity = localModel.doPredict(inputActivity)
        activityToList(outputActivity)
      })
    })
  }
}
