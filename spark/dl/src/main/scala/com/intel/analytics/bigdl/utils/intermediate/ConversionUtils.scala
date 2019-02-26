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

package com.intel.analytics.bigdl.utils.intermediate

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.mkldnn.DnnGraph
import com.intel.analytics.bigdl.utils.{Engine, MklDnn, T}
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.StaticGraph

import scala.reflect.ClassTag

private[bigdl] object ConversionUtils {
  /**
   * convert model to ir graph and build
   * @param model
   * @return
   */
  def convert[T: ClassTag](model: Module[T]): Module[T] = {
    if (model.isInstanceOf[IRGraph[T]]) {
      val g = model.asInstanceOf[IRGraph[T]]
      if (g.isBuild) g else g.build()
    } else if (!model.isInstanceOf[DnnGraph] && Engine.getEngineType() == MklDnn) {
      val m = if (!model.isInstanceOf[Graph[T]]) model.toGraph() else model
      if (!m.isInstanceOf[StaticGraph[T]]) return model
      val ir = m.asInstanceOf[StaticGraph[T]].toIRgraph().asInstanceOf[Module[T]]
      if (model.isTraining()) ir.training() else ir.evaluate()
      ir
    } else {
      model
    }
  }

  /**
   * For dnn backend, it is recommended to run single model on each node.
   * So when partition number of dataset is not equal to node number,
   * there will be coalesce operation.
   * @param dataset
   * @tparam T
   * @return
   */
  def coalesce[T: ClassTag](dataset: RDD[T]): RDD[T] = {
    if (dataset.getNumPartitions != Engine.nodeNumber()
      && Engine.getEngineType() == MklDnn) {
      dataset.coalesce(Engine.nodeNumber(), false)
    } else dataset
  }
}
