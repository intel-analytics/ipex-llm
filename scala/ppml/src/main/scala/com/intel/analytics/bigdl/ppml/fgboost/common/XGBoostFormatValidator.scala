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

package com.intel.analytics.bigdl.ppml.fgboost.common

import org.apache.log4j.LogManager


class XGBoostFormatValidator {

}
object XGBoostFormatValidator {
  val logger = LogManager.getLogger(getClass)
  def apply(treeArray1: Array[XGBoostFormatNode], treeArray2: Array[XGBoostFormatNode]) = {
    def validateTreeEquality(t1: XGBoostFormatNode, t2: XGBoostFormatNode): Boolean = {
      def almostEqual(a: Float, b: Float) = {
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
        require(t1.children.tail.size == 1 && t2.children.tail.size == 1, "???")
        // validate meta info
        if (t1.split != t2.split || t1.depth != t2.depth) {
          logger.error(s"t1->split:${t1.split}, depth:${t1.depth}, " +
            s"t2->split:${t2.split}, depth:${t2.depth}")
          return false
        }
        validateTreeEquality(t1.children.head, t2.children.head) &&
          validateTreeEquality(t1.children.tail.head, t2.children.tail.head)
      }
    }
    require(treeArray1.length == treeArray1.length, "Boosting round not equal, " +
      s"FGBoost: ${treeArray1.length}, XGBoost: ${treeArray2.length}")
    // TODO: validate first 15 trees for now
    treeArray1.slice(0, 15).indices.foreach(i => {
      require(validateTreeEquality(treeArray1(i), treeArray2(i)), s"Tree $i not same")
    })
  }
}
