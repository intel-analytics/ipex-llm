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

package com.intel.analytics.bigdl.ppml.fl.fgboost.common

import org.apache.log4j.LogManager

import scala.collection.mutable.ArrayBuffer
import com.intel.analytics.bigdl.dllib.utils.Log4Error

class XGBoostFormatValidator {

}
object XGBoostFormatValidator {
  val logger = LogManager.getLogger(getClass)
  var fGBoostHeaders: ArrayBuffer[Array[String]] = new ArrayBuffer[Array[String]]()
  var xGBoostHeaders: Array[String] = _
  def setXGBoostHeaders(headers: Array[String]): Unit = {
    xGBoostHeaders = headers
  }
  def clearHeaders(): Unit = {
    fGBoostHeaders.clear()
  }
  def addHeaders(headers: Array[String]): Unit = {
    fGBoostHeaders.append(headers)
  }

  /**
   * To validate two trees in [[XGBoostFormatNode]] format.
   * Note that the order of two args could not be swapped.
   * XGBoost is a non-distributed algorithm and it would not record the feature names
   * FGBoost is distributed, so the feature ID would be re-ordered, will record feature names
   * @param treeArray1 The XGBoost result in format, the feature would be represented in int
   * @param treeArray2 The FGBoost result in format, the feature would be represented in string
   */
  def apply(treeArray1: Array[XGBoostFormatNode], treeArray2: Array[XGBoostFormatNode]): Unit = {
    def validateTreeEquality(t1: XGBoostFormatNode, t2: XGBoostFormatNode): Boolean = {
      def almostEqual(a: Float, b: Float): Boolean = {
        math.abs(a - b) < 1e-3
      }
      if (t1.children == null) {
        if (t2.children != null) {
          logger.error("t2 not null ???")
          return false
        }
        if (almostEqual(t1.leaf, t2.leaf)) true
        else {
          logger.error(s"t1->leaf:${t1.leaf}, t2->leaf:${t2.leaf}")
          false
        }
      } else {
        Log4Error.unKnowExceptionError(t1.children.tail.size == 1 && t2.children.tail.size == 1,
          "???")
        // validate meta info
        val xGBoostFeature = xGBoostHeaders(t1.split.toInt)
        var matchFlag = false
        fGBoostHeaders.foreach(onePartyFeature => {
          if (t2.split < onePartyFeature.length && onePartyFeature(t2.split) == xGBoostFeature) {
            matchFlag = true
          }
        })
        if (!matchFlag) {
          logger.error(s"could not find ${xGBoostFeature} in all party features, features:")
          fGBoostHeaders.foreach(onePartyFeature => {
            if (t2.split < onePartyFeature.length) {
              logger.error(onePartyFeature(t2.split))
            }
          })
          return false
        }

        if (t1.depth != t2.depth) {
          logger.error(s"t1 depth:${t1.depth}, " +
            s"t2- depth:${t2.depth}")
          return false
        }
        validateTreeEquality(t1.children.head, t2.children.head) &&
          validateTreeEquality(t1.children.tail.head, t2.children.tail.head)
      }
    }
    Log4Error.unKnowExceptionError(treeArray1.length == treeArray1.length,
      "Boosting round not equal, " +
      s"FGBoost: ${treeArray1.length}, XGBoost: ${treeArray2.length}")
    // TODO: validate first 15 trees for now
    treeArray1.slice(0, 15).indices.foreach(i => {
      Log4Error.unKnowExceptionError(validateTreeEquality(treeArray1(i), treeArray2(i)),
        s"Tree $i not same")
    })
  }
}
