package com.intel.analytics.dllib.lib.tensor

import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.dllib.lib.tensor.TensorType.{DoubleReal => Real}
import com.github.fommil.netlib.BLAS.{getInstance => blas}

object SparseTensorMath {

  //    def addmv(beta : Real, alpha : Real, mat : Tensor[Real], vec2 : Tensor[Real]) : Tensor[Real] = {
    def addmv[@specialized(Float, Double) T](r : Tensor[T], beta : T, t : Tensor[T], alpha : T,
                                             mat : Tensor[T], vec : Tensor[T])(implicit ev:TensorNumeric[T]) : Tensor[T] = {
    require(mat.nDimension() == 2 && vec.nDimension() == 1)
    require(mat.size(2) == vec.size(1))
    require(t.nDimension() == 1)
    require(t.size(1) == mat.size(1))
    if(!r.eq(t)) {
      r.resizeAs(t).copy(t)
    }

    (alpha, mat, vec, beta, r)  match {
      case (alpha: Double, a:SparseTensor[Double], x:DenseTensor[Double], beta:Double, y:DenseTensor[Double]) =>
        SparseTensorBLAS.coodgemv(alpha, a, x, beta, y)
      case (alpha: Float, a:SparseTensor[Float], x:DenseTensor[Float], beta:Float, y:DenseTensor[Float]) =>
        SparseTensorBLAS.coosgemv(alpha, a, x, beta, y)
      case (alpha: Double, a:SparseTensorCsr[Double], x:DenseTensor[Double], beta:Double, y:DenseTensor[Double]) =>
        SparseTensorBLAS.csrdgemv(alpha, a, x, beta, y)
      case _ =>
        throw new IllegalArgumentException(s"Sparse addmv doesn't support")
    }
    r
  }
}
