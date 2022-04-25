/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.fgboost

import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer
import com.intel.analytics.bigdl.ppml.example.DebugLogger
import com.intel.analytics.bigdl.ppml.fgboost.common.XGBoostFormatNode
import ml.dmlc.xgboost4j.scala.DMatrix
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class XGBoostSpec extends FlatSpec with Matchers with BeforeAndAfter with DebugLogger{
  "XGBoost Sparse Dmatrix from file" should "work" in {
    val mat = new DMatrix(getClass.getClassLoader
      .getResource("xgboost/xgboost-sparse.txt").getPath)
    mat
  }
  "XGBoost model dump json single node" should "work" in {
    val dataPath = getClass.getClassLoader.getResource("xgboost/single-node.json").getPath
    val jsonStr = scala.io.Source.fromFile(dataPath).mkString
    val jacksonJsonSerializer = new JacksonJsonSerializer()
    val a = jacksonJsonSerializer.deSerialize(classOf[XGBoostFormatNode], jsonStr)
    a
  }
  "XGBoost model dump json small tree" should "work" in {
    val dataPath = getClass.getClassLoader.getResource("xgboost/small-tree.json").getPath
    val jsonStr = scala.io.Source.fromFile(dataPath).mkString
    val jacksonJsonSerializer = new JacksonJsonSerializer()
    val a = jacksonJsonSerializer.deSerialize(classOf[XGBoostFormatNode], jsonStr)
    a

  }
}
