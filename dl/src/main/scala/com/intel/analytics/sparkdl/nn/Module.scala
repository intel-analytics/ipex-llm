/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.Tensor
import org.apache.commons.lang3.SerializationUtils

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

abstract class Module[T: ClassTag](implicit ev: TensorNumeric[T]) extends Serializable {
  var output: Tensor[T] = Tensor[T]()
  var gradInput: Tensor[T] = Tensor[T]()

  var gradWeight: Tensor[T] = null
  var gradBias: Tensor[T] = null
  var gradient: (Tensor[T], Tensor[T]) = (gradWeight, gradBias)

  private var name : String = null

  def setName(name : String) : this.type = {
    this.name = name
    this
  }

  def getName() : String = {
    if (this.name == null) this.toString else this.name
  }

  // list of sub modules
  val modules: ArrayBuffer[Module[T]] = ArrayBuffer[Module[T]]()

  protected var train: Boolean = true

  protected var forwardTime = 0L

  protected var backwardTime = 0L

  def getTimes(): Array[(Module[T], Long, Long)] = {
    Array((this, forwardTime, backwardTime))
  }

  def resetTimes(): Unit = {
    forwardTime = 0
    backwardTime = 0
  }

  final def forward(input: Tensor[T]): Tensor[T] = {
    val before = System.nanoTime()
    val result = updateOutput(input)
    forwardTime += System.nanoTime() - before
    result
  }

  def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val before = System.nanoTime()
    val result = updateGradInput(input, gradOutput)
    accGradParameters(input, gradOutput)
    backwardTime += System.nanoTime() - before
    result
  }

  def updateOutput(input: Tensor[T]): Tensor[T] = {
    this.output = input
    input
  }

  def updateOutput(input: Tensor[T], flag: Int): Tensor[T] = {
    this.output = input
    input
  }

  def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T]

  def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double = 1.0): Unit = {}

  def zeroGradParameters(): Unit = {}

  def updateParameters(learningRate: T): Unit = {}

  def getParameters(): (Tensor[T], Tensor[T]) = {
    val (weightParameters, gradParameters) = this.parameters()
    return (Module.flatten(weightParameters), Module.flatten(gradParameters))
  }

  /**
   * @return (Array of weights, Array of grad)
   */
  def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = null

  def training(): this.type = {
    train = true
    this
  }

  /**
   * Find a module by given a parameter offset
   *
   * @param paramOffset parameter offset in the (weight, grad) vector returned by the
   *                    getParamter function
   * @param indexes     ignore it
   * @return module ref, offset(ignore), indexes from the current module
   */
  def findModel(paramOffset: Int,
    indexes: Array[Int] = Array()): (Module[T], Int, Array[Int]) = (this, paramOffset, indexes)

  def mapModules(f: Module[T] => Unit): Unit = {
    f(this)

    if (modules != null) {
      modules.foreach(_.mapModules(f))
    }
  }

  def findModules(name: String): ArrayBuffer[Module[T]] = {
    def matchName(module: Module[T]) =
      module.getClass.getName.equals(name)

    val nodes = new ArrayBuffer[Module[T]]()

    if (matchName(this)) nodes.append(this)
    if (modules != null) {
      modules.foreach(m => {
        val tempNodes = m.findModules(name)
        nodes ++= tempNodes
      })
    }

    nodes
  }



  def evaluate(): this.type = {
    train = false
    this
  }

  final def isTraining(): Boolean = {
    this.train
  }

  def reset(): Unit = {}

  protected var line = "\n"

  def setLine(line: String): this.type = {
    this.line = line
    this
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[Module[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Module[T]]
    if (this.eq(other)) {
      return true
    }
    if (output != other.output) {
      return false
    }
    if (gradInput != other.gradInput) {
      return false
    }
    if (gradWeight == null) {
      if (other.gradWeight != null) {
        return false
      }
    } else {
      if (gradWeight != other.gradWeight) {
        return false
      }
    }
    if (gradBias == null) {
      if (other.gradBias != null) {
        return false
      }
    } else {
      if (gradBias != other.gradBias) {
        return false
      }
    }

    true
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = 1
    if (output != null) {
      hash = hash * seed + this.output.hashCode()
    }
    if (gradInput != null) {
      hash = hash * seed + this.gradInput.hashCode()
    }
    if (gradWeight != null) {
      hash = hash * seed + this.gradWeight.hashCode()
    }
    if (gradBias != null) {
      hash = hash * seed + this.gradBias.hashCode()
    }

    hash
  }

  def cloneModule(): Module[T] = {
    SerializationUtils.clone(this)
  }
}

object Module {
  def flatten[@specialized(Float, Double) T: ClassTag](paramters: Array[Tensor[T]])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    val compactedTensor = isCompact(paramters)
    if (compactedTensor != null) {
      return compactedTensor
    }
    var i = 0
    var length = 0
    while (i < paramters.length) {
      require(paramters(i).isContiguous())
      length += paramters(i).nElement()
      i += 1
    }

    val result = Tensor[T](length)
    val resultStorage = result.storage()

    i = 0
    var offset = 0
    while (i < paramters.length) {
      System.arraycopy(paramters(i).storage().array(), paramters(i).storageOffset() - 1,
        resultStorage.array(), offset, paramters(i).nElement())
      paramters(i).set(resultStorage, offset + 1, paramters(i).size(), paramters(i).stride())
      offset += paramters(i).nElement()
      i += 1
    }

    result
  }

  def isCompact[@specialized(Float, Double) T: ClassTag](paramters: Array[Tensor[T]])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(paramters.length > 0)
    var i = 1
    val storage = paramters(0).storage()
    var length = paramters(0).nElement()
    while (i < paramters.length) {
      if (!storage.eq(paramters(i).storage())) {
        return null
      }
      length += paramters(i).nElement()
      i += 1
    }

    if (length != storage.array().length) {
      return null
    }

    return Tensor(storage)
  }
}




