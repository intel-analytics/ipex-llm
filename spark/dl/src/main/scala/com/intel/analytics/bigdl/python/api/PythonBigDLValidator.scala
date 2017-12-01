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

package com.intel.analytics.bigdl.python.api

import java.lang.{Boolean => JBoolean}
import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.collection.JavaConverters._
import scala.collection.mutable.Map
import scala.language.existentials
import scala.reflect.ClassTag

object PythonBigDLValidator {

  def ofFloat(): PythonBigDLValidator[Float] = new PythonBigDLValidator[Float]()

  def ofDouble(): PythonBigDLValidator[Double] = new PythonBigDLValidator[Double]()
}

class PythonBigDLValidator[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T]{

  def testDict(): JMap[String, String] = {
    return Map("jack" -> "40", "lucy" -> "50").asJava
  }

  def testDictJTensor(): JMap[String, JTensor] = {
    return Map("jack" -> JTensor(Array(1.0f, 2.0f, 3.0f, 4.0f), Array(4, 1), "float")).asJava
  }

  def testDictJMapJTensor(): JMap[String, JMap[String, JTensor]] = {
    val table = new Table()
    val tensor = JTensor(Array(1.0f, 2.0f, 3.0f, 4.0f), Array(4, 1), "float")
    val result = Map("jack" -> tensor).asJava
    table.insert(tensor)
    return Map("nested" -> result).asJava
  }

  def testActivityWithTensor(): JActivity = {
    val tensor = Tensor(Array(1.0f, 2.0f, 3.0f, 4.0f), Array(4, 1))
    return JActivity(tensor)
  }

  def testActivityWithTableOfTensor(): JActivity = {
    val tensor1 = Tensor(Array(1.0f, 1.0f), Array(2))
    val tensor2 = Tensor(Array(2.0f, 2.0f), Array(2))
    val tensor3 = Tensor(Array(3.0f, 3.0f), Array(2))
    val table = new Table()
    table.insert(tensor1)
    table.insert(tensor2)
    table.insert(tensor3)
    return JActivity(table)
  }

  def testActivityWithTableOfTable(): JActivity = {
    val tensor = Tensor(Array(1.0f, 2.0f, 3.0f, 4.0f), Array(4, 1))
    val table = new Table()
    table.insert(tensor)
    val nestedTable = new Table()
    nestedTable.insert(table)
    nestedTable.insert(table)
    return JActivity(nestedTable)
  }
}
