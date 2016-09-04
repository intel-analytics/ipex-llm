package com.intel.analytics.dllib.lib.nn

import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.dllib.lib.tensor.{torch, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration.Duration
import scala.concurrent.{ExecutionContext, Await, Future}
import scala.reflect._

class SpatialMaxPooling[@specialized(Float, Double) T: ClassTag](val kW: Int, val kH: Int, val dW: Int, val dH: Int, val padW: Int = 0, val padH: Int = 0)
                                                      (implicit ev: TensorNumeric[T]) extends Module[T]{

  var ceil_mode = false
  var indices = torch.Tensor[T]()

  def this(kW: Int, kH: Int)(implicit ev: TensorNumeric[T]){
    this(kW, kH, kW, kH)
  }

  def ceil(): SpatialMaxPooling[T] = {
    ceil_mode = true
    this
  }

  def floor(): SpatialMaxPooling[T] = {
    ceil_mode = false
    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim()==3 || input.dim() == 4, "3D or 4D (batch mode) tensor expected")
    val dimw = input.dim()
    val dimh = input.dim() - 1
    require(input.size(dimw) >= kW - padW && input.size(dimh) >= kH - padH, "input image smaller than kernel size")
    require(kW/2 >= padW && kH/2 >= padH, "pad should be smaller than half of kernel size")
    val nslices = input.size(dimh - 1)
    val iheight = input.size(dimh)
    val iwidth = input.size(dimw)
    var oheight:Int = 0
    var owidth:Int = 0
    if(ceil_mode){
      oheight = math.ceil(1.0*(iheight - kH + 2*padH)/dH).toInt + 1
      owidth = math.ceil(1.0*(iwidth - kW + 2*padW)/dW).toInt + 1
    }
    else {
      oheight = math.floor(1.0*(iheight - kH + 2*padH)/dH).toInt + 1
      owidth = math.floor(1.0*(iwidth - kW + 2*padW)/dW).toInt + 1
    }

    if(padW != 0 || padH != 0) {
      if((oheight - 1) * dH >= iheight + padH) oheight -= 1
      if((owidth - 1) * dW >= iwidth + padW) owidth -= 1
    }

    if(input.dim() == 3){
      output.resize(Array(nslices,oheight,owidth))
      /* indices will contain the locations for each output point */
      indices.resize(Array(nslices,oheight,owidth))
      if(classTag[T] == classTag[Double]) {
        NNPrimitive.maxPoolingForwardDouble(
          input.asInstanceOf[Tensor[Double]].storage().array(), input.storageOffset() - 1,
          output.asInstanceOf[Tensor[Double]].storage().array(), output.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Double]].storage().array(), indices.storageOffset() - 1,
          nslices, iwidth, iheight, owidth, oheight, kW, kH, dW, dH, padW, padH)
      } else if(classTag[T] == classTag[Float]) {
        NNPrimitive.maxPoolingForwardFloat(
          input.asInstanceOf[Tensor[Float]].storage().array(), input.storageOffset() - 1,
          output.asInstanceOf[Tensor[Float]].storage().array(), output.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Float]].storage().array(), indices.storageOffset() - 1,
          nslices, iwidth, iheight, owidth, oheight, kW, kH, dW, dH, padW, padH)
      } else
        ???
    }
    else {
      val nbatch = input.size(1)
      output.resize(Array(nbatch, nslices, oheight, owidth))
      indices.resize(Array(nbatch, nslices, oheight, owidth))
      val results = new ArrayBuffer[Future[Unit]]()
      if(classTag[T] == classTag[Double]) {
        for (i <- 1 to nbatch) results += Future {
          val curInput = input(i)
          val curOutput = output(i)
          val curIndices = indices(i)
          NNPrimitive.maxPoolingForwardDouble(
            curInput.asInstanceOf[Tensor[Double]].storage().array(), curInput.storageOffset() - 1,
            curOutput.asInstanceOf[Tensor[Double]].storage().array(), curOutput.storageOffset() - 1,
            curIndices.asInstanceOf[Tensor[Double]].storage().array(), curIndices.storageOffset() - 1,
            nslices, iwidth, iheight, owidth, oheight,
            kW, kH, dW, dH, padW, padH)
        }
      } else if(classTag[T] == classTag[Float]) {
        for (i <- 1 to nbatch) results += Future {
          val curInput = input(i)
          val curOutput = output(i)
          val curIndices = indices(i)
          NNPrimitive.maxPoolingForwardFloat(
            curInput.asInstanceOf[Tensor[Float]].storage().array(), curInput.storageOffset() - 1,
            curOutput.asInstanceOf[Tensor[Float]].storage().array(), curOutput.storageOffset() - 1,
            curIndices.asInstanceOf[Tensor[Float]].storage().array(), curIndices.storageOffset() - 1,
            nslices, iwidth, iheight, owidth, oheight,
            kW, kH, dW, dH, padW, padH)
        }
      } else
        ???

      for(t <- results) {
        Await.result(t, Duration.Inf)
      }
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dimw = input.dim()
    val dimh = input.dim() - 1
    require(input.size(dimw) >= kW - padW && input.size(dimh) >= kH - padH, "input image smaller than kernel size")
    require(kW/2 >= padW && kH/2 >= padH, "pad should be smaller than half of kernel size")
    val nslices = input.size(dimh - 1)
    val iheight = input.size(dimh)
    val iwidth = input.size(dimw)
    val oheight:Int = gradOutput.size(dimh)
    val owidth:Int = gradOutput.size(dimw)
    gradInput.resizeAs(input)
    gradInput.zero()
    if(input.dim() == 3){
      if(classTag[T] == classTag[Double]) {
        NNPrimitive.maxPoolingBackwardDouble(
          gradInput.asInstanceOf[Tensor[Double]].storage().array(), gradInput.storageOffset() - 1,
          gradOutput.asInstanceOf[Tensor[Double]].storage().array(), gradOutput.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Double]].storage().array(), indices.storageOffset() - 1,
          nslices, iwidth, iheight, owidth, oheight)
      } else if(classTag[T] == classTag[Float]) {
        NNPrimitive.maxPoolingBackwardFloat(
          gradInput.asInstanceOf[Tensor[Float]].storage().array(), gradInput.storageOffset() - 1,
          gradOutput.asInstanceOf[Tensor[Float]].storage().array(), gradOutput.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Float]].storage().array(), indices.storageOffset() - 1,
          nslices, iwidth, iheight, owidth, oheight)
      } else {
        ???
      }
    }
    else {
      val nbacth = input.size(1)
      val results = new ArrayBuffer[Future[Unit]]()
      if(classTag[T] == classTag[Double]) {
        for(k <- 1 to nbacth) results += Future {
          val curGradInput = gradInput(k)
          val curGradOutput = gradOutput(k)
          val curIndices = indices(k)
          NNPrimitive.maxPoolingBackwardDouble(
            curGradInput.asInstanceOf[Tensor[Double]].storage().array(), curGradInput.storageOffset() - 1,
            curGradOutput.asInstanceOf[Tensor[Double]].storage().array(), curGradOutput.storageOffset() - 1,
            curIndices.asInstanceOf[Tensor[Double]].storage().array(), curIndices.storageOffset() - 1,
            nslices, iwidth, iheight, owidth, oheight)
        }
      } else if(classTag[T] == classTag[Float]) {
        for (k <- 1 to nbacth) results += Future  {
          val curGradInput = gradInput(k)
          val curGradOutput = gradOutput(k)
          val curIndices = indices(k)
          NNPrimitive.maxPoolingBackwardFloat(
            curGradInput.asInstanceOf[Tensor[Float]].storage().array(), curGradInput.storageOffset() - 1,
            curGradOutput.asInstanceOf[Tensor[Float]].storage().array(), curGradOutput.storageOffset() - 1,
            curIndices.asInstanceOf[Tensor[Float]].storage().array(), curIndices.storageOffset() - 1,
            nslices, iwidth, iheight, owidth, oheight)
        }
      } else {
        ???
      }

      for(t <- results) {
        Await.result(t, Duration.Inf)
      }
    }
    gradInput
  }

  override def equals(obj : Any) : Boolean = {

    if(!super.equals(obj)) {
      return false
    }

    if(!obj.isInstanceOf[SpatialMaxPooling[T]])
      return false
    val other = obj.asInstanceOf[SpatialMaxPooling[T]]
    if(this.eq(other))
      return true

    kW == other.kW &&
      kH == other.kH &&
      dW == other.dW &&
      dH == other.dH &&
      padW == other.padW &&
      padH == other.padH &&
      ceil_mode ==other.ceil_mode &&
      indices == other.indices
  }

  override def toString() : String = {
    s"nn.SpatialMaxPooling($kW, $kH, $dW, $dH, $padW, $padH)"
  }
}
