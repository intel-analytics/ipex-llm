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

package com.intel.analytics.bigdl.dllib.feature

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.feature.common.{Relation, RelationPair, Relations}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class RelationsSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val path: String = getClass.getClassLoader.getResource("qa").getPath
  val txtRelations: String = path + "/relations.txt"
  val csvRelations: String = path + "/relations.csv"
  val parquetRelations: String = path + "/relations.parquet"
  val targetRelations = Set(Relation("Q1", "A1", 1), Relation("Q1", "A2", 0),
    Relation("Q2", "A1", 0), Relation("Q2", "A2", 1))
  var sc : SparkContext = _

  before {
    val conf = new SparkConf().setAppName("Test read relations").setMaster("local[1]")
    sc = NNContext.initNNContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "Read txt file with sc" should "work properly" in {
    val relations = Relations.read(txtRelations, sc)
    require(relations.count() == 4)
    require(relations.collect().toSet == targetRelations)
  }

  "Read txt file without sc" should "work properly" in {
    val relations = Relations.read(txtRelations)
    require(relations.length == 4)
    require(relations.toSet == targetRelations)
  }

  "Read csv file with sc" should "work properly" in {
    val relations = Relations.read(csvRelations, sc)
    require(relations.count() == 4)
    require(relations.collect().toSet == targetRelations)
  }

  "Read csv file without sc" should "work properly" in {
    val relations = Relations.read(csvRelations)
    require(relations.length == 4)
    require(relations.toSet == targetRelations)
  }

  "Read parquet file" should "work properly" in {
    val relations = Relations.readParquet(parquetRelations, SQLContext.getOrCreate(sc))
    require(relations.count() == 4)
    require(relations.collect().toSet == targetRelations)
  }

  "Generate relation pairs" should "work properly" in {
    val relations = Relations.read(csvRelations, sc)
    val relationPairs = Relations.generateRelationPairs(relations)
    require(relationPairs.count() == 2)
    require(relationPairs.collect().toSet == Set(RelationPair("Q1", "A1", "A2"),
      RelationPair("Q2", "A2", "A1")))
  }

}
