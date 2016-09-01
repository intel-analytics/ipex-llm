package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.torch
import org.scalatest.FlatSpec

import scala.math._

class SparseLinearSpec extends FlatSpec {
  "Linear module" should "converate to correct weight and bias" in {
    val m: Int = 3
    val a: Array[Double] = Array(2.0, 5.0, 4.0, 6.0)
    val indices = Array(Array(1, 1, 2, 3), Array(1, 2, 2, 3))
    val target = torch.Tensor(torch.storage[Double](Array(12.0, 8.0, 18.0)))

    val mat = torch.Tensor(indices, torch.storage[Double](a), Array(3, 3))

    val sl = new SparseLinear[Double](3, 1)
    sl.weight(Array(1)) = 1
    sl.weight(Array(2)) = 2
    sl.weight(Array(3)) = 3
    sl.bias(Array(1)) = 0.5
    val output = sl.forward(mat)
    sl.backward(mat, target)
    println(output)


    val denMat = torch.Tensor[Double](3, 3)
    denMat(Array(1, 1)) = 2
    denMat(Array(1, 2)) = 5
    denMat(Array(2, 2)) = 4
    denMat(Array(3, 3)) = 6
    val linear = new Linear[Double](3, 1)
    linear.weight(Array(1, 1)) = 1
    linear.weight(Array(1, 2)) = 2
    linear.weight(Array(1, 3)) = 3
    linear.bias(Array(1)) = 0.5
    val output2 = linear.forward(denMat)
    target.resize(Array(3, 1))
    linear.backward(denMat, target)
    println(output2)

  }


}
