package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric
import com.intel.webscaleml.nn.tensor.{SparseTensorMath, torch, Tensor}
import scala.reflect.ClassTag
import scala.util.Random

class SparseLinear[@specialized(Float, Double) T: ClassTag] (inputSize: Int,outputSize:Int)(implicit ev: TensorNumeric[T]) extends Module [T]{
  require(outputSize == 1, "outputSize should be 1 only")

  val weight: Tensor[T] = torch.Tensor[T](inputSize)
  val bias: Tensor[T] = torch.Tensor[T](outputSize)
  val addBuffer: Tensor[T] = torch.Tensor[T]()
  this.gradWeight = torch.Tensor[T](inputSize)
  this.gradBias = torch.Tensor[T](outputSize)
  reset()

  override def reset(): Unit ={
    val stdv = 1.0 /math.sqrt(weight.size(1))
    val random = new Random()
    weight.apply1(_=>ev.fromType[Double](random.nextDouble()*2*stdv - stdv)) //todo, better to support uniform
    bias.apply1(_=>ev.fromType[Double](random.nextDouble()*2*stdv - stdv))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")
    output.resize(Array(input.size(1)))
    output.fill(bias(Array(1)))
    SparseTensorMath.addmv(output, ev.fromType[Int](1), output, ev.fromType[Int](1), input, weight)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
      //TODO: implement updateGradInput
      gradInput
    }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double = 1.0): Unit ={
    SparseTensorMath.addmv(gradWeight, ev.fromType(0.0), gradWeight, ev.fromType[Double](scale), input.t(), gradOutput)
    gradBias.add(ev.times(ev.fromType[Double](scale), gradOutput.sum()))
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

    if(!obj.isInstanceOf[SparseLinear[T]])
      return false
    val other = obj.asInstanceOf[SparseLinear[T]]
    if(this.eq(other))
      return true

    gradWeight == other.gradWeight &&
      gradBias == other.gradBias &&
      weight == other.weight &&
      bias == other.bias
  }

  override def toString() : String = {
    s"nn.SparseLinear($inputSize -> $outputSize)"
  }

  override def findModel(paramOffset : Int, indexes : Array[Int]) : (Module[T], Int, Array[Int]) = {
    (this, paramOffset - outputSize * inputSize - outputSize, indexes)
  }
}
