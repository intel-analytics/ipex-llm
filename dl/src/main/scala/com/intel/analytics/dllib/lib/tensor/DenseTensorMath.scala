package com.intel.analytics.dllib.lib.tensor

import com.intel.analytics.dllib.lib.tensor.{DenseTensorApply => Apply}
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future, ExecutionContext}
import scala.reflect.ClassTag
import ExecutionContext.Implicits.global

object DenseTensorMath{
  val taskSize : Int = System.getProperty("cpu.task.size", "250000").toInt

  def mul[@specialized(Float, Double) T](self : DenseTensor[T], x : Tensor[T], value : T)
                                        (implicit ev:TensorNumeric[T]): Tensor[T] = {
    if(x != null) {
      self.copy(x)
    }

//    Apply.apply1[T](self, (d, i) => d(i) = ev.times(d(i), value))
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit =
      {data(index) = ev.times(data(index), value) }}
        Apply.apply1[T](self, func)
//    val data = self.storage().array
//    Apply.apply4(self, (i) => data(i)=ev.times(data(i), value))
    self
  }

  def div[@specialized(Float, Double) T](self : DenseTensor[T], x : Tensor[T], value : T)
                                        (implicit ev:TensorNumeric[T]): Tensor[T] = {
    if (x != null) {
      self.copy(x)
    }

    if(self.isContiguous()) {
      val data = self.storage().array()
      val tasks = for(taskOffset <- 0 until self.nElement() / taskSize + 1) yield Future {
        var i = taskOffset * taskSize + self.storageOffset() - 1
        while (i < self.nElement() && i < (taskOffset + 1) * taskSize) {
          data(i) = ev.divide(data(i), value)
          i += 1
        }
      }

      for(t <- tasks) {
        Await.result(t, Duration.Inf)
      }

    } else {
      val func = new TensorFunc2[T] {
        override def apply(data: Array[T], index: Int): Unit = {
          data(index) = ev.divide(data(index), value)
        }
      }
      Apply.apply1[T](self, func)
    }
    self
  }

  def cmul[@specialized(Float, Double) T](self : DenseTensor[T], y : Tensor[T])
                                         (implicit ev:TensorNumeric[T]): Tensor[T] = {
    require(self.nElement() == y.nElement(), "element number doesn't match")
//    Apply.apply2[T](self, y, (a, i1, b, i2) => a(i1) = ev.times(a(i1), b(i2)))
    val func2 = new TensorFunc4[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit =
      { data1(offset1) = ev.times(data2(offset2), data1(offset1)) }}
    Apply.apply2[T](self, y, func2)
    self
  }

  def cadd[@specialized(Float, Double) T](self : DenseTensor[T], x : Tensor[T], value : T, y : Tensor[T])
         (implicit ev:TensorNumeric[T]): Tensor[T] = {
    require(x != null)

    if(!self.eq(x)) {
      self.resizeAs(x).copy(x)
    }

    if(self.eq(x) && self.isContiguous() && y.isContiguous() && self.nElement() == y.nElement()) {
      ev.axpy(y.nElement(), value, y.storage().array(), y.storageOffset() - 1, 1, self.storage().array(), self.storageOffset() - 1, 1)
    } else {
//      Apply.apply2[T](self, y, (a, i1, b, i2) => a(i1) = ev.plus(a(i1), ev.times(value, b(i2))))
      val func2 = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit =
        { data1(offset1) = ev.plus(data1(offset1), ev.times(value, data2(offset2)))  }}
      Apply.apply2[T](self, y, func2)
    }
    self
  }

  def add[@specialized(Float, Double) T: ClassTag](s : T, t : DenseTensor[T])
                                        (implicit ev:TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(t)
    result.copy(t)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit =
      {data(index) = ev.plus(data(index),s)}}
    Apply.apply1[T](result, func)

    result
  }

  def add[@specialized(Float, Double) T: ClassTag](self : DenseTensor[T], t : Tensor[T])
                                        (implicit ev:TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(self)
    result.copy(self)
    val n = result.nElement()
    if(result.isContiguous() && t.isContiguous() && n == t.nElement()) {
      ev.axpy(n, ev.fromType[Int](1), t.storage().array(), t.storageOffset() - 1, 1,
        result.storage().array,
        result.storageOffset() - 1, 1)
      result
    } else {
      val func2 = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit =
        { data1(offset1) = ev.plus(data1(offset1), data2(offset2))  }}
      Apply.apply2[T](self, t, func2)
      result
    }
  }

  def sub[@specialized(Float, Double) T: ClassTag](s : T, t : DenseTensor[T])
                                        (implicit ev:TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(t)
    result.copy(t)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit =
      {data(index) = ev.minus(data(index), s) }}
        Apply.apply1[T](result, func)
    result
  }

  def sub[@specialized(Float, Double) T: ClassTag](self : DenseTensor[T], t : Tensor[T])
                                        (implicit ev:TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(self)
    result.copy(self)
    val func2 = new TensorFunc4[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit =
      { data1(offset1) = ev.minus(data1(offset1), data2(offset2))  }}
    Apply.apply2[T](result, t, func2)
    result
  }

  def neg[@specialized(Float, Double) T: ClassTag](self : DenseTensor[T])
         (implicit ev:TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(self)
    result.copy(self)

    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit =
      {data(index) = ev.negative(data(index)) }}
        Apply.apply1[T](result, func)
    result
  }
  def divide[@specialized(Float, Double) T: ClassTag](s : T, t : DenseTensor[T])
                                           (implicit ev:TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(t)
    result.copy(t)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit =
      {data(index) = ev.divide(data(index), s) }}
    Apply.apply1[T](result, func)
    result
  }
  def divide[@specialized(Float, Double) T: ClassTag](self : DenseTensor[T], t : Tensor[T])
            (implicit ev:TensorNumeric[T]) : Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(self)
    result.copy(self)
    val func2 = new TensorFunc4[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit =
      { data1(offset1) = ev.divide(data1(offset1), data2(offset2))  }}
    Apply.apply2[T](result, t, func2)
    result
  }

  def mul[@specialized(Float, Double) T: ClassTag](s : T, t : DenseTensor[T])
                                        (implicit ev:TensorNumeric[T]): Tensor[T] = {
    val result = new DenseTensor[T]()
    result.resizeAs(t)
    result.copy(t)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit =
      {data(index) = ev.times(data(index), s) }}
    Apply.apply1[T](result, func)
    result
  }

  def mul[@specialized(Float, Double) T:ClassTag](self : Tensor[T], t : Tensor[T])
                                        (implicit ev:TensorNumeric[T]): Tensor[T] = {
    if(self.nDimension() == 1 && t.nDimension() == 1) {
      require(self.size(1) == t.size(1), "vector size not match")

      val result = ev.dot(self.nElement(), self.storage().array(), self.storageOffset() - 1, self.stride(1),
        t.storage().array(), t.storageOffset() - 1, t.stride(1))
      new DenseTensor(new ArrayStorage(Array(result)))
    } else if(self.nDimension() == 2 && t.nDimension() == 1) {
      val result = new DenseTensor[T](self.size(1))
      DenseTensorBLAS.dgemv[T](ev.fromType[Int](1), self, t, ev.fromType[Int](0), result)
      result
    } else if(self.nDimension() == 2 && t.nDimension() == 2){
      val result = new DenseTensor[T](t.size(2), self.size(1)).t()
      addmm[T](result, ev.fromType[Int](0), result, ev.fromType[Int](1), self, t)
      result
    } else{
      throw new UnsupportedOperationException(s"multiplication between ${self.nDimension()}D and " +
        s"${t.nDimension()}D not yet supported")
    }
  }

  def sumAll[@specialized(Float, Double) T](self : DenseTensor[T])(implicit ev:TensorNumeric[T]) : T = {
    var sum = ev.fromType[Int](0)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit =
      {sum = ev.plus(data(index), sum) }}
    Apply.apply1[T](self, func)
    sum
  }

  def sum[@specialized(Float, Double) T: ClassTag](self : DenseTensor[T], _dim : Int)(implicit ev:TensorNumeric[T]) : Tensor[T] = {
    require(_dim >= 0 && _dim < self.nDimension, s"dimension ${_dim + 1} out of range")
    val result = new DenseTensor[T]()
    val sizes = self.size()
    sizes(_dim) = 1
    DenseTensor.resize(result, sizes)
    DenseTensorDimApply.dimApply2[T](result, self, _dim,
      (rData, rOffset, rStride, rSize, tData, tOffset, tStride, tSize) => {
        var sum = ev.fromType[Int](0)
        var i = 0
        while(i < tSize) {
          sum = ev.plus(sum, tData(tOffset + i * tStride))
          i += 1
        }
        rData(rOffset) = sum
      })

    result
  }

  def maxAll[@specialized(Float, Double) T](self : DenseTensor[T])(implicit ev:TensorNumeric[T]) : T = {
    var max = ev.fromType[Int](0)
    var first = true
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        if(first){
          first = false
          max = data(index)
        } else if(ev.isGreater(data(index), max)) {
          max = data(index)}
      }}
    Apply.apply1[T](self, func)
    max
  }

  def addmm[@specialized(Float, Double) T: ClassTag](r : Tensor[T], beta : T, t : Tensor[T], alpha : T, m1 : Tensor[T], m2 : Tensor[T])
                                                    (implicit ev:TensorNumeric[T]): Tensor[T] = {
    require(m1.dim() == 2 && m2.dim() == 2, s"matrices expected, got ${m1.dim()}, ${m2.dim()} tensors")
    require(m1.size(2) == m2.size(1), s"size mismatch, m1:${m1.size().mkString("x")} m2:${m2.size().mkString("x")}")
    require(t.dim() == 2, s"matrix expected, got ${t.dim()} tensor for t")
    require(t.size(1) == m1.size(1) && t.size(2) == m2.size(2), s"size mismatch. t:${t.size().mkString("x")}, " +
      s"m1:${m1.size().mkString("x")} + m2:${m2.size().mkString("x")}")

    if(r != t) {
      r.resizeAs(t)
      r.copy(t)
    }

    var _r : Tensor[T] = null
    var _m1 : Tensor[T] = m1
    var _m2 : Tensor[T] = m2
    var transpose_r = ""
    if(r.stride(1) == 1 && r.stride(2) != 0) {
      transpose_r = "n"
      _r = r
    } else if(r.stride(2) == 1 && r.stride(1) != 0){
      val swap = _m2
      _m2 = _m1
      _m1 = swap
      transpose_r = "t"
      _r = r
    } else {
      transpose_r = "n"
      _r = new DenseTensor[T](r.size(2), r.size(1))
      _r.copy(r)
      _r = _r.transpose(1, 2)
    }

    val index1 = if(transpose_r == "n") 1 else 2
    val index2 = if(transpose_r == "n") 2 else 1
    var transpose_m1 = ""
    var __m1 : Tensor[T] = null
    if(_m1.stride(index1) == 1 && _m1.stride(index2) != 0) {
      transpose_m1 = "n"
      __m1 = _m1
    } else if(_m1.stride(index2) == 1 && _m1.stride(index1) != 0){
      transpose_m1 = "t"
      __m1 = _m1
    } else {
      transpose_m1 = if(transpose_r == "n") "t" else "n"
      __m1 = _m1.contiguous()
    }

    var transpose_m2 = ""
    var __m2 : Tensor[T] = null
    if(_m2.stride(index1) == 1 && _m2.stride(index2) != 0) {
      transpose_m2 = "n"
      __m2 = _m2
    } else if(_m2.stride(index2) == 1 && _m2.stride(index1) != 0){
      transpose_m2 = "t"
      __m2 = _m2
    } else {
      transpose_m2 = if(transpose_r == "n") "t" else "n"
      __m2 = _m2.contiguous()
    }

    DenseTensorBLAS.dgemm[T](transpose_m1, transpose_m2, _r.size(index1), _r.size(index2), __m1.size(index2), alpha,
      __m1.storage().asInstanceOf[Storage[T]].array(), __m1.storageOffset() - 1,
      if(transpose_m1 == "n") __m1.stride(index2) else __m1.stride(index1),
      __m2.storage().asInstanceOf[Storage[T]].array(), __m2.storageOffset() - 1,
      if(transpose_m2 == "n") __m2.stride(index2) else __m2.stride(index1),
      beta,
      _r.storage().asInstanceOf[Storage[T]].array(), _r.storageOffset() - 1,
      _r.stride(index2)
    )
    if(_r != r)
      r.copy(_r)
    r
  }

  def addr[@specialized(Float, Double) T](r : Tensor[T], beta : T, t : Tensor[T],
           alpha : T, vec1 : Tensor[T], vec2 : Tensor[T])(implicit ev:TensorNumeric[T]) : Tensor[T] = {
    require(vec1.dim() == 1 && vec2.dim() == 1)
    require(t.dim() == 2)
    require(t.size(1) == vec1.size(1) && t.size(2) == vec2.size(1))

    if(!r.eq(t)) {
      r.resizeAs(t).copy(t)
    }

    if(beta != 1) {
      r.mul(beta)
    }

    if(r.stride(1) == 1) {
      val lda = if(t.stride(2) == 1) {
        r.size(1)
      } else {
        r.stride(2)
      }
      ev.ger(vec1.size(1), vec2.size(1), alpha, vec1.storage().asInstanceOf[Storage[T]].array(), vec1.storageOffset() - 1, vec1.stride(1),
        vec2.storage().asInstanceOf[Storage[T]].array(), vec2.storageOffset() - 1, vec2.stride(1), r.storage().asInstanceOf[Storage[T]].array(), r.storageOffset() - 1, lda)
    } else if(r.stride(2) == 1) {
      ev.ger(vec2.size(1), vec1.size(1), alpha, vec2.storage().asInstanceOf[Storage[T]].array(), vec2.storageOffset() - 1, vec2.stride(1),
        vec1.storage().asInstanceOf[Storage[T]].array(), vec1.storageOffset() - 1, vec1.stride(1), r.storage().asInstanceOf[Storage[T]].array(), r.storageOffset() - 1, r.stride(1))
    } else {
      val cr = r.contiguous()
      ev.ger(vec2.size(1), vec1.size(1), alpha, vec2.storage().asInstanceOf[Storage[T]].array(), vec2.storageOffset() - 1, vec2.stride(1),
        vec1.storage().asInstanceOf[Storage[T]].array(), vec1.storageOffset() - 1, vec1.stride(1), cr.storage().asInstanceOf[Storage[T]].array(), cr.storageOffset() - 1, cr.stride(1))
      r.copy(cr)
    }

    r
  }

  def addmv[@specialized(Float, Double) T](r : Tensor[T], beta : T, t : Tensor[T], alpha : T,
           mat : Tensor[T], vec : Tensor[T])(implicit ev:TensorNumeric[T]) : Tensor[T] = {
    require(mat.nDimension() == 2 && vec.nDimension() == 1)
    require(mat.size(2) == vec.size(1))
    require(t.nDimension() == 1)
    require(t.size(1) == mat.size(1))

    if(!r.eq(t)) {
      r.resizeAs(t).copy(t)
    }

    if(mat.stride(1) == 1) {
      ev.gemv("N", mat.size(1), mat.size(2), alpha, mat.storage().asInstanceOf[Storage[T]].array(), mat.storageOffset() - 1,
        mat.stride(2),vec.storage().asInstanceOf[Storage[T]].array(), vec.storageOffset() - 1, vec.stride(1), beta, r.storage().asInstanceOf[Storage[T]].array(),
        r.storageOffset() - 1, r.stride(1))
    } else if(mat.stride(2) == 1) {
      ev.gemv("T", mat.size(2), mat.size(1), alpha, mat.storage().asInstanceOf[Storage[T]].array(), mat.storageOffset() - 1,
        mat.stride(1),vec.storage().asInstanceOf[Storage[T]].array(), vec.storageOffset() - 1, vec.stride(1), beta, r.storage().asInstanceOf[Storage[T]].array(),
        r.storageOffset() - 1, r.stride(1))
    } else {
      val cmat = mat.contiguous()
      ev.gemv("T", cmat.size(2), cmat.size(1), alpha, cmat.storage().asInstanceOf[Storage[T]].array(), cmat.storageOffset() - 1,
        cmat.stride(1),vec.storage().asInstanceOf[Storage[T]].array(), vec.storageOffset() - 1, vec.stride(1), beta, r.storage().asInstanceOf[Storage[T]].array(),
        r.storageOffset() - 1, r.stride(1))
    }

    r
  }

  def meanAll[@specialized(Float, Double) T](self : DenseTensor[T])(implicit ev:TensorNumeric[T]) : T = {
    var sum = ev.fromType[Int](0)
//    DenseTensorApply.apply1[T](self, (data, index) => sum = ev.plus(sum, data(index)))
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit =
      {sum = ev.plus(data(index), sum) }}
    Apply.apply1[T](self, func)
//    val data = self.storage().array
//    DenseTensorApply.apply4(self, (i) => sum = ev.plus(sum, data(i)))
    ev.divide(sum, ev.fromType[Int](self.nElement()))
  }

  def mean[@specialized(Float, Double) T:ClassTag](self : DenseTensor[T], _dim : Int)(implicit ev:TensorNumeric[T]) : Tensor[T] = {
    require(_dim >= 0 && _dim < self.nDimension, s"dimension ${_dim + 1} out of range")
    val result = new DenseTensor[T]()
    val sizes = self.size()
    sizes(_dim) = 1
    DenseTensor.resize(result, sizes)
    DenseTensorDimApply.dimApply2[T](result, self, _dim,
      (rData, rOffset, rStride, rSize, tData, tOffset, tStride, tSize) => {
        var sum = ev.fromType[Int](0)
        var i = 0
        while(i < tSize) {
          sum = ev.plus(sum, tData(tOffset + i * tStride))
          i += 1
        }
        rData(rOffset) = ev.divide(sum , ev.fromType[Int](self.size(_dim + 1)))
    })
    result
  }

  def nearlyEqual[@specialized(Float, Double) T](a : T, b : T, epsilon : Double)(implicit ev:TensorNumeric[T]) : Boolean = {
    ev.getType() match {
      case "Float" =>
        val floatA = ev.toType[Float](a)
        val floatB = ev.toType[Float](b)
        val absA = math.abs(floatA)
        val absB = math.abs(floatB)
        val diff = math.abs(floatA - floatB)

        val result = if (floatA == floatB) {
          true
        } else if (floatA == 0 || floatB == 0 || diff < java.lang.Float.MIN_NORMAL) {
          diff < (epsilon * java.lang.Float.MIN_NORMAL)
        } else {
          diff / (absA + absB) < epsilon
        }

        if(!result) {
          if (floatA == b) {
            true
          } else if (floatA == 0 || floatB == 0 || diff < java.lang.Float.MIN_NORMAL) {
            diff < (epsilon * java.lang.Float.MIN_NORMAL)
          } else {
            diff / (absA + absB) < epsilon
          }
        }
        result
      case "Double" =>
        val doubleA = ev.toType[Double](a)
        val doubleB = ev.toType[Double](b)
        val absA = math.abs(doubleA)
        val absB = math.abs(doubleB)
        val diff = math.abs(doubleA - doubleB)

        val result = if (doubleA == doubleB) {
          true
        } else if (doubleA == 0 || doubleB == 0 || diff < java.lang.Double.MIN_NORMAL) {
          diff < (epsilon * java.lang.Double.MIN_NORMAL)
        } else {
          diff / (absA + absB) < epsilon
        }

        if(!result) {
          if (doubleA == b) {
            true
          } else if (doubleA == 0 || doubleB == 0 || diff < java.lang.Double.MIN_NORMAL) {
            diff < (epsilon * java.lang.Double.MIN_NORMAL)
          } else {
            diff / (absA + absB) < epsilon
          }
        }
        result
    }
  }

  val doubleEpsilon = System.getProperty("DoubleTensorEpsilon", "0.0000001").toDouble
  val floatEpsilon = System.getProperty("FloatTensorEpsilon", "0.00001").toDouble
}

