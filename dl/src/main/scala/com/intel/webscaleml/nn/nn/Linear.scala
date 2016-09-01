package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric
import com.intel.webscaleml.nn.tensor.{torch, Tensor}
import scala.reflect.ClassTag
import com.intel.webscaleml.nn.tensor.RandomGenerator._

class Linear[@specialized(Float, Double) T: ClassTag](
  inputSize: Int,
  outputSize:Int,
  private var initMethod : InitializationMethod = Default
)(implicit ev: TensorNumeric[T]) extends Module[T]{
  val weight: Tensor[T] = torch.Tensor[T](outputSize,inputSize)
  val bias: Tensor[T] = torch.Tensor[T](outputSize)
  val addBuffer: Tensor[T] = torch.Tensor[T]()
  this.gradWeight = torch.Tensor[T](outputSize,inputSize)
  this.gradBias = torch.Tensor[T](outputSize)
  reset()

  def setInitMethod(initMethod : InitializationMethod) : this.type = {
    this.initMethod = initMethod
    this
  }

  override def reset(): Unit ={
    initMethod match {
      case Default =>
        val stdv = 1.0 /math.sqrt(weight.size(2))
        weight.apply1(_=> ev.fromType[Double](RNG.uniform(0,1)*2*stdv - stdv)) //todo, better to support uniform
        bias.apply1(_ => ev.fromType[Double](RNG.uniform(0,1)*2*stdv - stdv))
      case Xavier =>
        val fanIn = weight.size(2)
        val fanOut = weight.size(1)
        val stdv = math.sqrt(3 / (fanIn + fanOut))
        weight.apply1(_=>ev.fromType[Double](RNG.uniform(0,1)*2*stdv - stdv)) //todo, better to support uniform
        bias.fill(ev.fromType(0))
      case _ => ???
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] ={
    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")
    if(input.dim() == 1){
      output.resize(Array(outputSize))
      output.copy(bias)
      output.addmv(ev.fromType[Int](1), weight, input)
    }
    else if(input.dim() == 2) {
      val nFrame = input.size(1)
      val nElement = output.nElement
//      output.resize(Array(nFrame, bias.size(1)))
      val t = Array(nFrame, bias.size(1))
      output.resize(t)
      if(output.nElement() != nElement)
        output.zero()

      if(addBuffer.nElement() != nFrame)
        addBuffer.resize(Array(nFrame)).fill(ev.fromType[Int](1))

      output.addmm(ev.fromType[Int](0), output, ev.fromType[Int](1), input, weight.t)
      output.addr(ev.fromType[Int](1), addBuffer, bias)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] ={
    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")
    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if(nElement != gradInput.nElement()) {
      gradInput.zero()
    }

    if(input.dim() == 1) {
      gradInput.addmv(ev.fromType[Int](0), ev.fromType[Int](1), weight.t(), gradOutput)
    } else if(input.dim() == 2) {
      gradInput.addmm(ev.fromType[Int](0), ev.fromType[Int](1), gradOutput, weight)
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double = 1.0): Unit ={
    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")
    val value = ev.fromType[Double](scale)
    if(input.dim() == 1) {
      gradWeight.addr(value, gradOutput, input)
      gradBias.add(value, gradOutput)
    }
    else if(input.dim() == 2) {
      gradWeight.addmm(value, gradOutput.t, input)
      gradBias.addmv(value, gradOutput.t, addBuffer)
    }
  }

  override def updateParameters(learningRate:T): Unit ={
    //weight.map(gradWeight,(a,b)=>a - learningRate*b)
    weight.add(ev.negative(learningRate), gradWeight)
    //bias.map(gradBias,(a,b)=>a - learningRate*b)
    bias.add(ev.negative(learningRate), gradBias)
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override  def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) ={
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def equals(obj : Any) : Boolean = {

    if(!super.equals(obj)) {
      return false
    }

    if(!obj.isInstanceOf[Linear[T]])
      return false
    val other = obj.asInstanceOf[Linear[T]]
    if(this.eq(other))
      return true

    gradWeight == other.gradWeight &&
      gradBias == other.gradBias &&
      weight == other.weight &&
      bias == other.bias
  }

  override def toString() : String = {
    s"nn.Linear($inputSize -> $outputSize)"
  }

  override def findModel(paramOffset : Int, indexes : Array[Int]) : (Module[T], Int, Array[Int]) = {
    (this, paramOffset - outputSize * inputSize - outputSize, indexes)
  }

}
