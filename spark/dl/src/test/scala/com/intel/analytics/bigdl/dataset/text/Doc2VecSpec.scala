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

package com.intel.analytics.bigdl.dataset.text

import java.io.{PrintWriter}

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{FlatSpec, Matchers}

class Doc2VecSpec extends FlatSpec with Matchers {

  "Doc2VecSpec" should "vectorize articles correctly on Spark" in {

    val doc1 = "people have new president in the u.s."
    val doc2 = "Who is there?"
    val doc3 = "which state are you from?"
    val doc4 = "she's not that into you."

    val docs = Array(doc1, doc2, doc3, doc4)

    val v1 = "the 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457 -0.49688 -0.17862"
    val v2 = "is -0.00066023 -0.6566 0.27843 -0.14767 -0.55677 0.14658 -0.0095095 0.011658"
    val v3 = "are 0.10204 -0.12792 -0.8443 -0.12181 -0.016801 -0.33279 -0.1552 -0.23131"
    val v4 = "into -0.19181 -1.8823 -0.76746 0.099051 -0.42125 -0.19526 4.0071 -0.18594"
    val v5 = "that -0.52287 -0.31681 0.00059213 0.0074449 0.17778 -0.15897 0.012041 -0.054223"

    val dictS = v1 + "\n" + v2 + "\n" + v3 + "\n" + v4 + "\n" + v5

    val glove6BFilePathTemp = java.io.File
      .createTempFile("UnitTest", "Doc2VecSpec").getPath

    println(glove6BFilePathTemp)

    new PrintWriter(glove6BFilePathTemp) {
      write(dictS);
      close
    }

    Engine.init(1, 1, true)
    val conf = new SparkConf().setMaster("local[1]").setAppName("Doc2Vec")
    val sc = new SparkContext(conf)

    val vectors: DataSet[Array[Float]] = DataSet.rdd(sc.parallelize(docs))
      .transform(Doc2Vec(glove6BFilePathTemp))

    val output = vectors.toDistributed().data(train = false).collect()

    val numOfVectors = 4

    output.length should be(numOfVectors)

    sc.stop()
  }
}
