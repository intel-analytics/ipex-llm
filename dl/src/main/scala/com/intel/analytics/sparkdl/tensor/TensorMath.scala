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

package com.intel.analytics.sparkdl.tensor

/**
 * It provides multiple math operation functions for manipulating Tensor objects.
 * All functions support both allocating a new Tensor to return the result
 * and treating the caller as a target Tensor, in which case the target Tensor(s)
 * will be resized accordingly and filled with the result. This property is especially
 * useful when one wants to have tight control over when memory is allocated.
 *
 * @tparam T should be double or float
 */
trait TensorMath[T] {
  // scalastyle:off methodName

  /**
   * Add all elements of this with value not in place.
   * It will allocate new memory.
   * @param s
   * @return
   */

  def +(s: T): Tensor[T]

  /**
   * Add a Tensor to another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   * @param t
   * @return
   */
  def +(t: Tensor[T]): Tensor[T]

  def +(e: Either[Tensor[T], T]): Tensor[T] = {
    e match {
      case Right(scalar) => this + scalar
      case Left(tensor) => this + tensor
    }
  }

  /**
   * subtract all elements of this with the value not in place.
   * It will allocate new memory.
   * @param s
   * @return
   */
  def -(s: T): Tensor[T]

  /**
   * Subtract a Tensor from another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   * @param t
   * @return
   */
  def -(t: Tensor[T]): Tensor[T]

  def unary_-(): Tensor[T]

  /**
   * divide all elements of this with value not in place.
   * It will allocate new memory.
   * @param s
   * @return
   */
  def /(s: T): Tensor[T]

  /**
   * Divide a Tensor by another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   * @param t
   * @return
   */
  def /(t: Tensor[T]): Tensor[T]

  /**
   * multiply all elements of this with value not in place.
   * It will allocate new memory.
   * @param s
   * @return
   */
  def *(s: T): Tensor[T]

  /**
   * Multiply a Tensor by another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   * @param t
   * @return
   */
  def *(t: Tensor[T]): Tensor[T]

  // scalastyle:on methodName

  /**
   * returns the sum of the elements of this
   * @return
   */
  def sum(): T

  /**
   * performs the sum operation over the dimension dim
   * @param dim
   * @return
   */
  def sum(dim: Int): Tensor[T]

  def sum(x: Tensor[T], dim: Int): Tensor[T]

  /**
   * returns the mean of all elements of this.
   * @return
   */
  def mean(): T

  /**
   * performs the mean operation over the dimension dim.
   *
   * @param dim
   * @return
   */
  def mean(dim: Int): Tensor[T]

  /**
   * returns the single biggest element of x
   * @return
   */
  def max(): T

  /**
   * performs the max operation over the dimension n
   * @param dim
   * @return
   */
  def max(dim: Int): (Tensor[T], Tensor[T])

  /**
   * This function computes 2 dimensional convolution of a single image
   * with a single kernel (2D output). the dimensions of input and kernel
   * need to be 2, and Input image needs to be bigger than kernel. The
   * last argument controls if the convolution is a full ('F') or valid
   * ('V') convolution. The default is valid convolution.
   *
   * @param kernel
   * @param vf full ('F') or valid ('V') convolution.
   * @return
   */
  def conv2(kernel: Tensor[T], vf: Char = 'V'): Tensor[T]

  /**
   * This function operates with same options and input/output configurations as conv2,
   * but performs cross-correlation of the input with the kernel k.
   *
   * @param kernel
   * @param vf full ('F') or valid ('V') convolution.
   * @return
   */
  def xcorr2(kernel: Tensor[T], vf: Char = 'V'): Tensor[T]

  /**
   * replaces all elements in-place with the square root of the elements of this.
   * @return
   */
  def sqrt(): Tensor[T]

  /**
   * replaces all elements in-place with the absolute values of the elements of this.
   * @return
   */
  def abs(): Tensor[T]

  /**
   * x.add(value,y) multiply-accumulates values of y into x.
   *
   * @param value scalar
   * @param y     other tensor
   * @return current tensor
   */
  def add(value: T, y: Tensor[T]): Tensor[T]

  /**
   * accumulates all elements of y into this
   *
   * @param y other tensor
   * @return current tensor
   */
  def add(y: Tensor[T]): Tensor[T]

  // Puts the result of x + value * y in current tensor
  /**
   * z.add(x, value, y) puts the result of x + value * y in z.
   *
   * @param x
   * @param value
   * @param y
   * @return
   */
  def add(x: Tensor[T], value: T, y: Tensor[T]): Tensor[T]

  /**
   * x.add(value) : add value to all elements of x in place.
   * @param value
   * @return
   */
  def add(value: T): Tensor[T]

  def add(x: Tensor[T], y: Tensor[T]): Tensor[T]
  /**
   * Performs the dot product. The number of elements must match: both Tensors are seen as a 1D
   * vector.
   *
   * @param y
   * @return
   */
  def dot(y: Tensor[T]): T

  /**
   * Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the
   * scalar value (1 if not present) and add it to x. The number of elements must match, but sizes
   * do not matter.
   *
   * @param value
   * @param tensor1
   * @param tensor2
   */
  def addcmul(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T]

  def addcmul(tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T]

  /**
   * Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar
   * value and add it to x.
   * The number of elements must match, but sizes do not matter.
   *
   * @param value
   * @param tensor1
   * @param tensor2
   * @return
   */
  def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T]

  def sub(value : T, y : Tensor[T]) : Tensor[T]

  // Puts the result of x - value * y in current tensor
  def sub(x : Tensor[T], value : T, y : Tensor[T]) : Tensor[T]

  /**
   * subtracts all elements of y from this
   *
   * @param y other tensor
   * @return current tensor
   */
  def sub(y : Tensor[T]) : Tensor[T]

  def sub(x : Tensor[T], y : Tensor[T]) : Tensor[T]

  def sub(value : T) : Tensor[T]

  /**
   * Element-wise multiply
   * x.cmul(y) multiplies all elements of x with corresponding elements of y.
   * x = x * y
   *
   * @param y tensor
   * @return current tensor
   */
  def cmul(y: Tensor[T]): Tensor[T]

  /**
   * Element-wise multiply
   * z.cmul(x, y) equals z = x * y
   *
   * @param x tensor
   * @param y tensor
   * @return current tensor
   */
  def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T]

  /**
   * Element-wise divide
   * x.cdiv(y) all elements of x divide all elements of y.
   * x = x / y
   *
   * @param y tensor
   * @return current tensor
   */
  def cdiv(y: Tensor[T]): Tensor[T]

  /**
   * Element-wise divide
   * z.cdiv(x, y) means z = x / y
   *
   * @param x tensor
   * @param y tensor
   * @return current tensor
   */
  def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T]

  /**
   * multiply all elements of this with value in-place.
   *
   * @param value
   * @return
   */
  def mul(value: T): Tensor[T]

  /**
   * divide all elements of this with value in-place.
   *
   * @param value
   * @return
   */
  def div(value: T): Tensor[T]

  /**
   * put the result of x * value in current tensor
   *
   * @param value
   * @return
   */
  def mul(x: Tensor[T], value: T): Tensor[T]

  /**
   * Performs a matrix-matrix multiplication between mat1 (2D tensor) and mat2 (2D tensor).
   * Optional values v1 and v2 are scalars that multiply M and mat1 * mat2 respectively.
   * Optional value beta is a scalar that scales the result tensor, before accumulating the result
   * into the tensor. Defaults to 1.0.
   * If mat1 is a n x m matrix, mat2 a m x p matrix, M must be a n x p matrix.
   *
   * res = (v1 * M) + (v2 * mat1*mat2)
   *
   * @param v1
   * @param M
   * @param v2
   * @param mat1
   * @param mat2
   */
  def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T]

  // res = M + (mat1*mat2)
  def addmm(M: Tensor[T], mat1: Tensor[T], mat2: Tensor[T]): Tensor[T]

  // res = res + mat1 * mat2
  def addmm(mat1: Tensor[T], mat2: Tensor[T]): Tensor[T]

  // res = res + v2 * mat1 * mat2
  def addmm(v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T]

  // res = v1 * res + v2 * mat1*mat2
  def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T]

  /**
   * Performs the outer-product between vec1 (1D tensor) and vec2 (1D tensor).
   * Optional values v1 and v2 are scalars that multiply mat and vec1 [out] vec2 respectively.
   * In other words,
   * res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
   *
   * @param t1
   * @param t2
   * @return
   */
  def addr(t1: Tensor[T], t2: Tensor[T]): Tensor[T]

  def addr(v1: T, t1: Tensor[T], t2: Tensor[T]): Tensor[T]

  def addr(v1: T, t1: Tensor[T], v2: T, t2: Tensor[T]): Tensor[T]

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
   * Performs a matrix-vector multiplication between mat (2D Tensor) and vec2 (1D Tensor) and add
   * it to vec1. Optional values v1 and v2 are scalars that multiply vec1 and vec2 respectively.
   *
   * In other words,
   * res = (beta * vec1) + alpha * (mat * vec2)
   *
   * Sizes must respect the matrix-multiplication operation: if mat is a n × m matrix,
   * vec2 must be vector of size m and vec1 must be a vector of size n.
   */
  def addmv(beta: T, vec1: Tensor[T], alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T]

  // res = beta * res + alpha * (mat * vec2)
  def addmv(beta: T, alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T]

  // res = res + alpha * (mat * vec2)
  def addmv(alpha: T, mat: Tensor[T], vec2: Tensor[T]): Tensor[T]

  /**
   * Replaces all elements in-place with the elements of x to the power of n
   *
   * @param y
   * @param n
   * @return current tensor reference
   */
  def pow(y: Tensor[T], n : T): Tensor[T]

  def pow(n: T): Tensor[T]

  /**
   * Get the top k smallest values and their indices.
   *
   * @param result   result buffer
   * @param indices  indices buffer
   * @param k
   * @param dim      dimension, default is the last dimension
   * @param increase sort order, set it to true if you want to get the smallest top k values
   * @return
   */
  def topk(k: Int, dim: Int = -1, increase: Boolean = true, result: Tensor[T] = null,
    indices: Tensor[T] = null)
  : (Tensor[T], Tensor[T])

  /**
   * Replaces all elements in-place with the elements of lnx
   *
   * @param y
   * @return current tensor reference
   */
  def log(y: Tensor[T]): Tensor[T]

  def exp(y: Tensor[T]): Tensor[T]

  def sqrt(y: Tensor[T]): Tensor[T]

  def log1p(y: Tensor[T]): Tensor[T]

  def log(): Tensor[T]

  def exp(): Tensor[T]

  def log1p(): Tensor[T]


  def abs(x: Tensor[T]): Tensor[T]

  /**
   * Implements > operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  def gt(x: Tensor[T], y: Tensor[T]): Tensor[T]

  /**
   * mplements < operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  def lt(x: Tensor[T], y: Tensor[T]): Tensor[T]

  /**
   * mplements <= operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  def le(x: Tensor[T], y: Tensor[T]): Tensor[T]

  /**
   * Implements == operator comparing each element in x with y
   *
   * @param y
   * @return current tensor reference
   */
  def eq(x: Tensor[T], y: T): Tensor[T]

  /**
   * Fills the masked elements of itself with value val
   *
   * @param mask
   * @param e
   * @return current tensor reference
   */
  def maskedFill(mask: Tensor[T], e: T): Tensor[T]

  /**
   * Copies the elements of tensor into mask locations of itself.
   *
   * @param mask
   * @param y
   * @return current tensor reference
   */
  def maskedCopy(mask: Tensor[T], y: Tensor[T]): Tensor[T]

  /**
   * Returns a new Tensor which contains all elements aligned to a 1 in the corresponding mask.
   *
   * @param mask
   * @param y
   * @return current tensor reference
   */
  def maskedSelect(mask: Tensor[T], y: Tensor[T]): Tensor[T]

  /**
   * returns the sum of the n-norms on the Tensor x
   * @param value the n-norms
   * @return
   */
  def norm(value: Int): T

  /**
   * returns a new Tensor with the sign (+/- 1 or 0) of the elements of x.
   * @return
   */
  def sign(): Tensor[T]
}
