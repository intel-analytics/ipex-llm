package com.intel.webscaleml.nn.tensor

import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

trait TensorMath[T] {
  def + (s : T) : Tensor[T]
  def + (t : Tensor[T]) : Tensor[T]
  def + (e : Either[Tensor[T], T]) : Tensor[T] = {
    e match {
      case Right(scalar) => this + scalar
      case Left(tensor) => this + tensor
    }
  }
  def - (s : T) : Tensor[T]
  def - (t : Tensor[T]) : Tensor[T]
  def unary_-() : Tensor[T]
  def / (s : T) : Tensor[T]
  def / (t : Tensor[T]) : Tensor[T]
  def * (s : T) : Tensor[T]
  def * (t : Tensor[T]) : Tensor[T]
  def sum() : T
  def sum(dim : Int) : Tensor[T]
  def mean() : T
  def mean(dim : Int) : Tensor[T]
  def max() : T
  def max(dim : Int) : (Tensor[T], Tensor[T])
  def conv2(kernel : Tensor[T], vf : Char = 'V') : Tensor[T]
  def xcorr2(kernel : Tensor[T], vf : Char = 'V') : Tensor[T]
  def sqrt() : Tensor[T]
  def abs() : Tensor[T]

  /**
   * x.add(value,y) multiply-accumulates values of y into x.
   *
   * @param value scalar
   * @param y other tensor
   * @return current tensor
   */
  def add(value : T, y : Tensor[T]) : Tensor[T]

  // Puts the result of x + value * y in current tensor
  def add(x : Tensor[T], value : T, y : Tensor[T]) : Tensor[T]

  def add(value : T) : Tensor[T]

  /**
   * Performs the dot product. The number of elements must match: both Tensors are seen as a 1D vector.
   * @param y
   * @return
   */
  def dot(y : Tensor[T]) : T

  /**
   * Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value
   * (1 if not present) and add it to x. The number of elements must match, but sizes do not matter.
   *
   * @param value
   * @param tensor1
   * @param tensor2
   */
  def addcmul(value : T, tensor1 : Tensor[T], tensor2 : Tensor[T]) : Tensor[T]
  def addcmul(tensor1 : Tensor[T], tensor2 : Tensor[T]) : Tensor[T]

  /**
   * Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to x.
   * The number of elements must match, but sizes do not matter.
   *
   * @param value
   * @param tensor1
   * @param tensor2
   * @return
   */
  def addcdiv(value : T, tensor1 : Tensor[T], tensor2 : Tensor[T]) : Tensor[T]

  /**
   * accumulates all elements of y into this
   * @param y other tensor
   * @return current tensor
   */
  def add(y : Tensor[T]) : Tensor[T]

  /**
   * y.cmul(x) multiplies all elements of y with corresponding elements of x.
   *
   * @param y other tensor
   * @return current tensor
   */
  def cmul(y : Tensor[T]) : Tensor[T]

  /**
   * multiply all elements of this with value in-place.
   * @param value
   * @return
   */
  def mul(value : T) : Tensor[T]

  /**
   * divide all elements of this with value in-place.
   * @param value
   * @return
   */
  def div(value : T) : Tensor[T]

  /**
   * put the result of x * value in current tensor
   * @param value
   * @return
   */
  def mul(x : Tensor[T], value : T) : Tensor[T]

  /**
   * Performs a matrix-matrix multiplication between mat1 (2D tensor) and mat2 (2D tensor).
   * Optional values v1 and v2 are scalars that multiply M and mat1 * mat2 respectively.
   * Optional value beta is a scalar that scales the result tensor, before accumulating the result into the tensor.
   * Defaults to 1.0.
   * If mat1 is a n x m matrix, mat2 a m x p matrix, M must be a n x p matrix.
   *
   * res = (v1 * M) + (v2 * mat1*mat2)
   * @param v1
   * @param M
   * @param v2
   * @param mat1
   * @param mat2
   */
  def addmm(v1 : T, M : Tensor[T], v2 : T, mat1 : Tensor[T], mat2 : Tensor[T]) : Tensor[T]
  // res = M + (mat1*mat2)
  def addmm(M : Tensor[T], mat1 : Tensor[T], mat2 : Tensor[T]) : Tensor[T]
  // res = res + mat1 * mat2
  def addmm(mat1 : Tensor[T], mat2 : Tensor[T]) : Tensor[T]
  // res = res + v2 * mat1 * mat2
  def addmm(v2 : T, mat1 : Tensor[T], mat2 : Tensor[T]) : Tensor[T]
  // res = v1 * res + v2 * mat1*mat2
  def addmm(v1 : T, v2 : T, mat1 : Tensor[T], mat2 : Tensor[T]) : Tensor[T]

  /**
   * Performs the outer-product between vec1 (1D tensor) and vec2 (1D tensor).
   * Optional values v1 and v2 are scalars that multiply mat and vec1 [out] vec2 respectively.
   * In other words,
   *   res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
   * @param t1
   * @param t2
   * @return
   */
  def addr(t1 : Tensor[T], t2 : Tensor[T]) : Tensor[T]
  def addr(v1 : T, t1 : Tensor[T], t2 : Tensor[T]) : Tensor[T]
  def addr(v1 : T, t1 : Tensor[T], v2 : T, t2 : Tensor[T]) : Tensor[T]

  /**
   * return pseudo-random numbers, require 0<=args.length<=2
   * if args.length = 0, return [0, 1)
   * if args.length = 1, return [1, args(0)] or [args(0), 1]
   * if args.length = 2, return [args(0), args(1)]
   *
   * @param args
   */
  def uniform(args: T*): T

  /**
   * Performs a matrix-vector multiplication between mat (2D Tensor) and vec2 (1D Tensor) and add it to vec1.
   * Optional values v1 and v2 are scalars that multiply vec1 and vec2 respectively.
   *
   * In other words,
   *     res = (beta * vec1) + alpha * (mat * vec2)
   *
   * Sizes must respect the matrix-multiplication operation: if mat is a n × m matrix, vec2 must be vector of
   * size m and vec1 must be a vector of size n.
   */
  def addmv(beta : T, vec1 : Tensor[T], alpha : T, mat : Tensor[T], vec2 : Tensor[T]) : Tensor[T]
  // res = beta * res + alpha * (mat * vec2)
  def addmv(beta : T, alpha : T, mat : Tensor[T], vec2 : Tensor[T]) : Tensor[T]
  // res = res + alpha * (mat * vec2)
  def addmv(alpha : T, mat : Tensor[T], vec2 : Tensor[T]) : Tensor[T]
}
