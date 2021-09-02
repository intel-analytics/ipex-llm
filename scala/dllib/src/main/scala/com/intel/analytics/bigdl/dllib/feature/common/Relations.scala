/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.feature.common

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

import scala.collection.mutable.{Map => MMap, ArrayBuffer}
import scala.io.Source

object Relations {

  /**
   * Read relations from csv or txt file.
   * Each record is supposed to contain the following three fields in order:
   * id1(String), id2(String) and label(Integer).
   *
   * For csv file, it should be without header.
   * For txt file, each line should contain one record with fields separated by comma.
   *
   * @param path The path to the relations file, which can either be a local or disrtibuted file
   *             system (such as HDFS) path.
   * @param sc An instance of SparkContext.
   * @param minPartitions Integer. A suggestion value of the minimal partition number for input
   *                      texts. Default is 1.
   * @return RDD of [[Relation]].
   */
  def read(path: String, sc: SparkContext, minPartitions: Int = 1): RDD[Relation] = {
    sc.textFile(path, minPartitions).map(line => {
      val subs = line.split(",")
      Relation(subs(0), subs(1), subs(2).toInt)
    })
  }

  /**
   * Read relations from csv or txt file.
   * Each record is supposed to contain the following three fields in order:
   * id1(String), id2(String) and label(Integer).
   *
   * For csv file, it should be without header.
   * For txt file, each line should contain one record with fields separated by comma.
   *
   * @param path The local file path to the relations file.
   * @return Array of [[Relation]].
   */
  def read(path: String): Array[Relation] = {
    val src = Source.fromFile(path)
    src.getLines().toArray.map(line => {
      val subs = line.split(",")
      Relation(subs(0), subs(1), subs(2).toInt)
    })
  }

  /**
   * Read relations from parquet file.
   * Schema should be the following:
   * "id1"(String), "id2"(String) and "label"(Integer).
   *
   * @param path The path to the parquet file.
   * @param sqlContext An instance of SQLContext.
   * @return RDD of [[Relation]].
   */
  def readParquet(path: String, sqlContext: SQLContext): RDD[Relation] = {
    sqlContext.read.parquet(path).rdd.map(row => {
      val id1 = row.getAs[String]("id1")
      val id2 = row.getAs[String]("id2")
      val label = row.getAs[Int]("label")
      Relation(id1, id2, label)
    })
  }

  /**
   * Generate all [[RelationPair]]s from given [[Relation]]s.
   * Essentially, for each positive relation (id1 and id2 with label>0), it will be
   * paired with every negative relation of the same id1 (id2 with label=0).
   */
  def generateRelationPairs(relations: RDD[Relation]): RDD[RelationPair] = {
    val positive = relations.filter(_.label > 0).groupBy(_.id1)
    val negative = relations.filter(_.label == 0).groupBy(_.id1)
    positive.cogroup(negative).flatMap(x => {
      val posIDs = x._2._1.flatten.toArray.map(_.id2)
      val negIDs = x._2._2.flatten.toArray.map(_.id2)
      posIDs.flatMap(y => negIDs.map(z => RelationPair(x._1, y, z)))
    })
  }

  /**
   * generateRelationPairs for Relation array
   */
  def generateRelationPairs(relations: Array[Relation]): Array[RelationPair] = {
    val relSet: MMap[String, MMap[Int, ArrayBuffer[String]]] = MMap()
    val pairList: ArrayBuffer[RelationPair] = ArrayBuffer()
    for (relation <- relations) {
      if (! relSet.contains(relation.id1)) {
        val id2Array: ArrayBuffer[String] = ArrayBuffer()
        id2Array.append(relation.id2)
        relSet(relation.id1) = MMap(relation.label -> id2Array)
      }
      else {
        val labelMap = relSet(relation.id1)
        if (! labelMap.contains(relation.label)) {
          val id2Array: ArrayBuffer[String] = ArrayBuffer()
          id2Array.append(relation.id2)
          labelMap(relation.label) = id2Array
          relSet(relation.id1) = labelMap
        }
        else {
          labelMap(relation.label).append(relation.id2)
        }
      }
    }

    for ((id1, labelMap) <- relSet) {
      if (labelMap.contains(0) && labelMap.contains(1)) {
        val negatives = labelMap(0).toArray
        val positives = labelMap(1).toArray
        for (id2Positive <- positives) {
          for (id2Negative <- negatives) {
            val pair = RelationPair(id1, id2Positive, id2Negative)
            pairList.append(pair)
          }
        }
      }
    }
    pairList.toArray
  }
}

/**
 * It represents the relationship between two items.
 */
case class Relation(id1: String, id2: String, label: Int)

/**
 * A relation pair is made up of two relations of the same id1:
 * Relation(id1, id2Positive, label>0) [Positive Relation]
 * Relation(id1, id2Negative, label=0) [Negative Relation]
 */
case class RelationPair(id1: String, id2Positive: String, id2Negative: String)
