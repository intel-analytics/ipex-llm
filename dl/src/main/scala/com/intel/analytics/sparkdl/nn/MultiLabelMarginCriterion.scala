package com.intel.analytics.sparkdl.nn

import breeze.linalg.dim
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.T

import scala.annotation
import scala.annotation.target
import scala.reflect.ClassTag

/**
  * Created by zhangli on 16-11-18.
  */
/**
  * Creates a criterion that optimizes a multi-class multi-classification hinge loss
  * (margin-based loss) between input x (a 1D Tensor) and output y (which is a 1D Tensor of target class indices):
  */
class MultiLabelMarginCriterion[T: ClassTag]
(val sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  @transient
  private var isTarget:Tensor[T] = null

  var gradInput = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    if (null == isTarget) isTarget = Tensor[T]()
    var nframe: Int = 0
    var dim: Int = 0

    require(input.nDimension() == 1 || input.nDimension() == 2, "vector or matrix expected")
    if (input.nDimension() == 1) {
      nframe = 1
      dim = input.size(0) //???
      require(target.nDimension() == 1 && annotation.target.size(0) == dim, "inconsistent target size")
    } else {
      nframe = input.size(0)
      dim = input.size(1)
      require(target.nDimension() == 2 && target.size(0) ==nframe && target.size(1) == dim, "inconsistent target size")
    }

    //minall(target) //???
    //maxall(target) //???

    val _target = target.contiguous()
    val _input = input.contiguous()

    val input_data = _input.storage().array()
    val target_data = _target.storage().array()

    isTarget.resizeAs(target).zero()
    val isTarget_data = isTarget.storage().array()


    var sum = 0
    var t = 0
    while (t < nframe) {
      var ddt = 0
      var dt = 0
      while (ddt < dim) {
        val target_idx = target_data(ddt)
        if (ev.isGreater(ev.fromType(0), target_idx)) {
          //break???
        }
        isTarget_data(target_idx) = ev.fromType(1)
        ddt += 1
      }

      while (dt < dim) {
        val  target_idx = target_data(dt)
        if (ev.isGreater(ev.fromType(0), target_idx)) {
          //break???
        }

        val input_target = input_data(target_idx)
        var d = 0
        while (d < dim) {
          if (isTarget_data(d) == 0) {
            val z = 1 - input_target + input_data(d)
            if (z > 0) sum += z
          }
          d += 1
        }

        dt += 1
      }
      t += 1
    }

    sum = sum / dim

    if (sizeAverage) sum = sum / nframe

    output = ev.fromType(sum)
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 1 || input.nDimension() == 2, "vector or matrix expected")
    var nframe: Int = 0
    var dim: Int = 0

    if (input.nDimension() == 1) {
      nframe = 1
      dim = input.size(0) //???
      require(target.nDimension() == 1 && target.size(0) == dim, "inconsistent target size")
      require(isTarget.nDimension() == 1 && isTarget.size(0) == dim, "inconsistent isTarget size")
    } else {
      nframe = input.size(0)
      dim = input.size(1)
      require(target.nDimension() == 2 && target.size(0) ==nframe && target.size(1) == dim,
        "inconsistent target size")
      require(isTarget.nDimension() == 2 && isTarget.size(0) ==nframe && isTarget.size(1) == dim,
        "inconsistent isTarget size")
    }

    //THArgCheck(THTensor_(minall)(target) >= 0, 3, "target out of range");
    //THArgCheck(THTensor_(maxall)(target) <= dim, 3, "target out of range");

    //THArgCheck(THTensor_(minall)(isTarget) >= 0, 3, "isTarget out of range");
    //THArgCheck(THTensor_(maxall)(isTarget) <= 1, 3, "isTarget out of range");

    val _target = target.contiguous()
    val _input = input.contiguous()
    val _isTarget = isTarget.contiguous()

    val input_data = _input.storage().array()
    val target_data = _target.storage().array()
    val isTarget_data = _isTarget.storage().array()

    val g = if (sizeAverage)  1./(nframe*dim) else 1./(dim)

    gradInput.resizeAs(input).zero()
    val gradInput_data = gradInput.storage().array()

    var t = 0
    while (t < nframe) {
      var dt = 0
      while (dt < dim) {
        val target_idx = target_data(dt)
        if (target_idx < 0)
          break;

        val input_target = input_data(target_idx)
        var d = 0
        while (d < dim) {
          if (!isTarget_data(d))
          {
            val  z = 1 - input_target + input_data(d)
            if (z > 0)
            {
              gradInput_data(target_idx) -= g
              gradInput_data(d) += g
            }
          }
          d += 1
        }
        dt += 1
      }
      t += 1
    }

    gradInput
  }



  override def toString(): String = {
    s"nn.Bilinear($inputSize1, $inputSize2, $outputSize, $biasRes)"
  }

  override def equals(other: Any): Boolean = other match {
    case that: MultiLabelMarginCriterion[T] =>
      (that.eq(this)) &&
        sizeAverage == that.sizeAverage
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(sizeAverage)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}
