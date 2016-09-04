package com.intel.analytics.dllib.lib.tensor

import com.intel.analytics.dllib.lib.tensor.TensorType.DoubleReal
import org.scalatest.{Matchers, FlatSpec}

class SparseTensorMathSpec extends FlatSpec with Matchers {

  "addmv" should "return right result" in {
      val m: Int = 3
      val a: Array[Double] = Array(2.0, 5.0, 4.0, 6.0)
      val indices = Array(Array(1, 1, 2, 3), Array(1, 2, 2, 3))
      val x = torch.Tensor(new ArrayStorage[DoubleReal](Array(1.0, 2.0, 3.0)))
    val y = torch.Tensor(new ArrayStorage[DoubleReal](Array(1.0, 2.0, 3.0)))

    val mat = new SparseTensor(indices, new ArrayStorage[DoubleReal](a), Array(3, 3))
    SparseTensorMath.addmv(y, 1.0, y, 2.0, mat, x)

    y(Array(1)) should be (25)
    y(Array(2)) should be (18)
    y(Array(3)) should be (39)

    println(y)
  }

  "addmv with transposed SparseTensor" should "return right result" in {
    val m: Int = 3
    val a: Array[Double] = Array(2.0, 5.0, 4.0, 6.0)
    val indices = Array(Array(1, 1, 2, 3), Array(1, 2, 2, 3))
    val x = torch.Tensor(new ArrayStorage[DoubleReal](Array(1.0, 2.0, 3.0)))
    val y = torch.Tensor(new ArrayStorage[DoubleReal](Array(1.0, 2.0, 3.0)))

    val mat = new SparseTensor(indices, new ArrayStorage[DoubleReal](a), Array(3, 3)).t()
    SparseTensorMath.addmv(y, 1.0, y, 2.0, mat, x)

    y(Array(1)) should be (5)
    y(Array(2)) should be (28)
    y(Array(3)) should be (39)

    println(y)
  }
}
