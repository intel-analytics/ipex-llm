package com.intel.webscaleml.nn.nn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric
import com.intel.webscaleml.nn.tensor.{torch, Tensor}
import org.apache.commons.lang3.SerializationUtils

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

//abstract class Module[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T]) extends Serializable{
abstract class Module[ T: ClassTag](implicit ev: TensorNumeric[T]) extends Serializable{
  var output: Tensor[T] = torch.Tensor[T]()
  var gradInput: Tensor[T] = torch.Tensor[T]()

  var gradWeight: Tensor[T] = null
  var gradBias: Tensor[T] = null
  var gradient: (Tensor[T], Tensor[T]) = (gradWeight, gradBias)

  // list of sub modules
  val modules: ArrayBuffer[Module[T]] = ArrayBuffer[Module[T]]()

  protected var train : Boolean = true

  protected var forwardTime = 0L

  protected var backwardTime = 0L

  def getTimes() : Array[(Module[T], Long, Long)] = {
    Array((this, forwardTime, backwardTime))
  }

  def resetTimes() : Unit = {
    forwardTime = 0
    backwardTime = 0
  }

  final def forward(input: Tensor[T]): Tensor[T] = {
    val before = System.nanoTime()
    val result = updateOutput(input)
    forwardTime += System.nanoTime() - before
    result
  }

  def backward(input: Tensor[T], gradOutput: Tensor[T]) : Tensor[T] = {
    val before = System.nanoTime()
    val result = updateGradInput(input, gradOutput)
    accGradParameters(input, gradOutput)
    backwardTime += System.nanoTime() - before
    result
  }

  def updateOutput(input : Tensor[T]): Tensor[T] = {
    this.output = input
    input
  }

  def updateOutput(input : Tensor[T], flag : Int): Tensor[T] = {
    this.output = input
    input
  }

  def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T]

  def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double = 1.0): Unit = {}

  def zeroGradParameters(): Unit = {}

  def updateParameters(learningRate: T): Unit = { }

  def getParameters() : (Tensor[T], Tensor[T]) = {
    val (weightParameters, gradParameters) = this.parameters()
    return (Module.flatten(weightParameters), Module.flatten(gradParameters))
  }

  /**
   * @return (Array of weights, Array of grad)
   */
  def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = null

  def training() : this.type = {
    train = true
    this
  }
  /**
   * Find a module by given a parameter offset
   * @param paramOffset parameter offset in the (weight, grad) vector returned by the getParamter function
   * @param indexes ignore it
   * @return module ref, offset(ignore), indexes from the current module
   */
  def findModel(paramOffset : Int, indexes : Array[Int] = Array()) : (Module[T], Int, Array[Int]) = (this, paramOffset, indexes)

  def evaluate() : this.type = {
    train = false
    this
  }

  final def isTraining() : Boolean = {
    this.train
  }

  def reset(): Unit = {}

  protected var line = "\n"

  def setLine(line : String) : this.type = {
    this.line = line
    this
  }

  override def equals(obj : Any) : Boolean = {
    if(obj == null)
      return false
    if(!obj.isInstanceOf[Module[T]])
      return false
    val other = obj.asInstanceOf[Module[T]]
    if(this.eq(other))
      return true
    if(output != other.output) {
      return false
    }
    if(gradInput != other.gradInput) {
      return false
    }
    if(gradWeight == null) {
      if(other.gradWeight != null) {
        return false
      }
    } else {
      if(gradWeight != other.gradWeight) {
        return false
      }
    }
    if(gradBias == null) {
      if(other.gradBias != null) {
        return false
      }
    } else {
      if(gradBias != other.gradBias) {
        return false
      }
    }
    
    true
  }

  def cloneModule() : Module[T] = {
    SerializationUtils.clone(this)
  }
}

object Module {
  def flatten[@specialized(Float, Double) T: ClassTag](paramters : Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Tensor[T] = {
    val compactedTensor = isCompact(paramters)
    if(compactedTensor != null) {
      return compactedTensor
    }
    var i = 0
    var length = 0
    while(i < paramters.length) {
      require(paramters(i).isContiguous())
      length += paramters(i).nElement()
      i += 1
    }

    val result = torch.Tensor[T](length)
    val resultStorage = result.storage()

    i = 0
    var offset = 0
    while(i < paramters.length) {
      System.arraycopy(paramters(i).storage().array(), paramters(i).storageOffset() - 1, resultStorage.array(), offset, paramters(i).nElement())
      paramters(i).set(resultStorage, offset + 1, paramters(i).size(), paramters(i).stride())
      offset += paramters(i).nElement()
      i += 1
    }

    result
  }

  def isCompact[@specialized(Float, Double) T:ClassTag](paramters : Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Tensor[T] = {
    require(paramters.length > 0)
    var i = 1
    val storage = paramters(0).storage()
    var length = paramters(0).nElement()
    while(i < paramters.length) {
      if(!storage.eq(paramters(i).storage())) {
        return null
      }
      length += paramters(i).nElement()
      i += 1
    }

    if(length != storage.array().length) {
      return null
    }

    return torch.Tensor(storage)
  }
}




