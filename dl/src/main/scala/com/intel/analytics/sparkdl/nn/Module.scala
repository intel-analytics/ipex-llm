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
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.utils.{File, T, Table, Activities}
import org.apache.commons.lang3.SerializationUtils

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._
import com.intel.analytics.sparkdl.mkl.MKL


abstract class TensorModule[@specialized(Float, Double) T: ClassTag]
  (implicit ev: TensorNumeric[T]) extends Module[Tensor[T], Tensor[T], T]

/**
 * Module is the basic component of a neural network. It forward activities and backward gradients.
 * Modules can connect to others to construct a complex neural network.
 *
 * @tparam A Input data type
 * @tparam B Output data type
 * @tparam T Numeric type. Only support float/double now
 */
abstract class Module[A <: Activities: ClassTag, B <: Activities: ClassTag,
  @specialized(Float, Double) T: ClassTag](
  implicit ev: TensorNumeric[T]) extends Serializable {

  /**
   * The cached output. So we don't compute it again when need it
   */
  var output: B = Activities[B, T]().asInstanceOf[B]

  /**
   * The cached gradient of activities. So we don't compute it again when need it
   */
  var gradInput: A = Activities[A, T]().asInstanceOf[A]


  /**
   * Clear cached activities to save storage space or network bandwidth. Note that we use
   * Tensor.set to keep some information like tensor share
   *
   * The subclass should override this method if it allocate some extra resource, and call the
   * super.clearState in the override method
   * @return
   */
  def clearState() : this.type = {
    if (output.isInstanceOf[Tensor[T]]) {
      output.asInstanceOf[Tensor[T]].set()
    }

    if (gradInput.isInstanceOf[Tensor[T]]) {
      gradInput.asInstanceOf[Tensor[T]].set()
    }

    this
  }

  /**
   * Init some necessary member, e.g. gradientWeight. Override it if subclass need to do it.
   *
   * This method will be call after construction or load from file
   * @return
   */
  def setup() : this.type = {
    this
  }

  /**
   * The name of the module
   */
  private var name : String = null

  /**
   * Set the module name
   * @param name
   * @return
   */
  def setName(name : String) : this.type = {
    this.name = name
    this
  }

  /**
   * get the module name
   * @return
   */
  def getName() : String = {
    if (this.name == null) this.getClass.getName else this.name
  }

  private var needComputeBack = true

  def setNeedComputeBack(need: Boolean): this.type = {
    needComputeBack = need
    this
  }

  def isNeedComputeBack(): Boolean = {
    needComputeBack
  }

  protected var forwardTime = 0L

  protected var backwardTime = 0L

  def getTimes(): Array[(Module[_ <: Activities, _ <: Activities, T], Long, Long)] = {
    Array((this, forwardTime, backwardTime))
  }

  def resetTimes(): Unit = {
    forwardTime = 0
    backwardTime = 0
  }

  /**
   * Takes an input object, and computes the corresponding output of the module. After a forward,
   * the output state variable should have been updated to the new value.
   *
   * @param input input data
   * @return output data
   */
  final def forward(input: A): B = {
    val before = System.nanoTime()
    val result = updateOutput(input)
    forwardTime += System.nanoTime() - before
    result
  }

  /**
   * Performs a back-propagation step through the module, with respect to the given input. In
   * general this method makes the assumption forward(input) has been called before, with the same
   * input. This is necessary for optimization reasons. If you do not respect this rule, backward()
   * will compute incorrect gradients.
   *
   * @param input input data
   * @param gradOutput gradient of next layer
   * @return gradient corresponding to input data
   */
  def backward(input: A, gradOutput: B): A = {
    val before = System.nanoTime()
    val result = updateGradInput(input, gradOutput)
    accGradParameters(input, gradOutput)
    backwardTime += System.nanoTime() - before
    result
  }

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   * @param input
   * @return
   */
  def updateOutput(input: A): B = {
    this.output = input.asInstanceOf[B]
    output
  }

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   * @param input
   * @param gradOutput
   * @return
   */
  def updateGradInput(input: A, gradOutput: B): A

  /**
   * Computing the gradient of the module with respect to its own parameters. Many modules do not
   * perform this step as they do not have any parameters. The state variable name for the
   * parameters is module dependent. The module is expected to accumulate the gradients with
   * respect to the parameters in some variable.
   * @param input
   * @param gradOutput
   * @param scale
   */
  def accGradParameters(input: A, gradOutput: B, scale: Double = 1.0): Unit = {}

  /**
   * If the module has parameters, this will zero the accumulation of the gradients with respect
   * to these parameters. Otherwise, it does nothing.
   */
  def zeroGradParameters(): Unit = {}

  def updateParameters(learningRate: T): Unit = {}

  /**
   * This method compact all parameters and gradients of the model into two tensors. So it's easier
   * to use optim method
   *
   * @return
   */
  def getParameters(): (Tensor[T], Tensor[T]) = {
    val (weightParameters, gradParameters) = this.parameters()
    (Module.flatten[T](weightParameters), Module.flatten[T](gradParameters))
  }

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = null

  /**
   * Module status. It is useful for modules like dropout/batch normalization
   */
  protected var train: Boolean = true

  def training(): this.type = {
    train = true
    this
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

  def cloneModule(): Module[A, B, T] = {
    SerializationUtils.clone(this)
  }

  def canEqual(other: Any): Boolean = other.isInstanceOf[Module[A, B, T]]

  override def equals(other: Any): Boolean = other match {
    case that: Module[A, B, T] =>
      (that canEqual this) &&
        (that.getClass equals this.getClass) &&
        output == that.output &&
        gradInput == that.gradInput
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Object): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(output, gradInput, this.getClass)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  def save(path : String, overWrite: Boolean = false) : this.type = {
    this.clearState()
    File.save(this, path, overWrite)
    this.setup()
    this
  }

  // Support for mkl init.
  def getClassPtr() : Long = {0L}
  def getInputPtr() : Long = getClassPtr()
  def getOutputPtr() : Long = getClassPtr()
  var hasSet = false
  def initMkl(prevPtr: Long) : Unit = {
    println("I WANT TO SET THE PREV LAYOUT IN MODULE")
    if (prevPtr != 0 && this.getClassPtr() != 0 &&
        prevPtr != this.getClassPtr()) {
      ev.getType() match {
        case "Double" =>
          MKL.SetPrevDouble(prevPtr, this.getClassPtr())
        case "Float" =>
          MKL.SetPrevFloat(prevPtr, this.getClassPtr())
        case _ =>
          throw new UnsupportedOperationException(s"Only Float/Double support")
      }
    }
  }

  var isPrevMkl = false
  var isNextMKl = false

  private var prevPtr = 0L
  private var nextPtr = 0L

  def setPrevPtr(ptr : Long) = { prevPtr = ptr }
  def setNextPtr(ptr : Long) = { nextPtr = ptr }
  def getPrevPtr() : Long = prevPtr
  def getNextPtr() : Long = nextPtr

  var initForward = true
  var initBackward = true

  def updateMklOut(): Unit = {
    // If the layer uses mkl dnn api, the ptr (prevPtr and classPtr) will not equal to 0.
    // And of cause the previous ptr and current ptr will not equal to each other.
    //println("prev = " + getPrevPtr().toHexString + " " + this.getName() + "\tcurrent = " + getClassPtr().toHexString)
    if (getPrevPtr() != 0 && getClassPtr() != getPrevPtr()) {
      ev.getType() match {
        case "Double" =>
          MKL.SetPrevDouble(getPrevPtr(), getInputPtr())
        case "Float" =>
          MKL.SetPrevFloat(getPrevPtr(), getInputPtr())
        case _ =>
          throw new UnsupportedOperationException(s"Only Float/Double support")
      }
    }
  }

  def updateMklGradInput() : Unit = {
    //println("next = " + getNextPtr().toHexString + " " + this.getName() + "\tcurrent = " + getClassPtr().toHexString)
    if (getNextPtr() != 0 && getClassPtr() != getNextPtr()) {
      ev.getType() match {
        case "Double" =>
          MKL.SetNextDouble(getNextPtr(), getOutputPtr())
        case "Float" =>
          MKL.SetNextFloat(getNextPtr(), getOutputPtr())
        case _ =>
          throw new UnsupportedOperationException(s"Only Float/Double support")
      }
    }
  }
}

object Module {
  def load[A <: Activities: ClassTag, B <: Activities: ClassTag,
  @specialized(Float, Double) T: ClassTag](path : String) : Module[A, B, T] = {
    File.load[Module[A, B, T]](path).setup()
  }

  def flatten[@specialized(Float, Double) T: ClassTag](parameters: Array[Tensor[T]])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    val compactedTensor = isCompact(parameters)
    if (compactedTensor != null) {
      return compactedTensor
    }
    var i = 0
    var length = 0
    while (i < parameters.length) {
      require(parameters(i).isContiguous())
      length += parameters(i).nElement()
      i += 1
    }

    val result = Tensor[T](length)
    val resultStorage = result.storage()

    i = 0
    var offset = 0
    while (i < parameters.length) {
      System.arraycopy(parameters(i).storage().array(), parameters(i).storageOffset() - 1,
        resultStorage.array(), offset, parameters(i).nElement())
      parameters(i).set(resultStorage, offset + 1, parameters(i).size(), parameters(i).stride())
      offset += parameters(i).nElement()
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




