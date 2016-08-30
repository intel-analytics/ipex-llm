package com.intel.webscaleml.nn.tensor

import java.io.Serializable
import breeze.linalg.{DenseMatrix => BrzDenseMatrix, DenseVector => BrzDenseVector}
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.mllib.linalg.{DenseVector, Vector, DenseMatrix, Matrix}

import scala.reflect.ClassTag

object torch {
  def Tensor[@specialized(Float, Double) T: ClassTag]()(implicit ev: TensorNumeric[T]) : Tensor[T] = new DenseTensor[T]()
  def Tensor[@specialized(Float, Double) T: ClassTag](d1 : Int)(implicit ev: TensorNumeric[T]) : Tensor[T] = new DenseTensor[T](d1)
  def Tensor[@specialized(Float, Double) T: ClassTag](d1 : Int, d2 : Int)(implicit ev: TensorNumeric[T]) : Tensor[T] = new DenseTensor[T](d1, d2)
  def Tensor[@specialized(Float, Double) T: ClassTag](d1 : Int, d2 : Int, d3 : Int)(implicit ev: TensorNumeric[T]) : Tensor[T] = new DenseTensor[T](d1, d2, d3)
  def Tensor[@specialized(Float, Double) T: ClassTag](d1 : Int, d2 : Int, d3 : Int, d4 : Int)(implicit ev: TensorNumeric[T]) : Tensor[T] = new DenseTensor[T](d1, d2, d3, d4)
  def Tensor[@specialized(Float, Double) T: ClassTag](d1 : Int, d2 : Int, d3 : Int, d4 : Int, d5 : Int)(implicit ev: TensorNumeric[T]) : Tensor[T] = new DenseTensor[T](d1, d2, d3, d4, d5)
  def Tensor[@specialized(Float, Double) T: ClassTag](dims : Int *)(implicit ev: TensorNumeric[T]) : Tensor[T] =
    new DenseTensor[T](new ArrayStorage[T](new Array[T](dims.product)), 0, dims.toArray, DenseTensor.size2Stride(dims.toArray), dims.length)
  def Tensor[@specialized(Float, Double) T: ClassTag](sizes : Array[Int])(implicit ev: TensorNumeric[T]) : Tensor[T] =
    new DenseTensor(new ArrayStorage[T](new Array[T](sizes.product)), 0, sizes.clone(), DenseTensor.size2Stride(sizes.clone()), sizes.length)
  def Tensor[@specialized(Float, Double) T: ClassTag](storage : Storage[T])(implicit ev: TensorNumeric[T]) : Tensor[T] = {
    new DenseTensor(storage.asInstanceOf[Storage[T]])
  }
  def Tensor[@specialized(Float, Double) T: ClassTag](storage : Storage[T], storageOffset : Int, size : Array[Int] = null, stride : Array[Int] = null)
                                                     (implicit ev: TensorNumeric[T]): Tensor[T] = {
    new DenseTensor(storage.asInstanceOf[Storage[T]], storageOffset, size, stride)
  }
  def Tensor[@specialized(Float, Double) T: ClassTag](other : Tensor[T])(implicit ev: TensorNumeric[T]) : Tensor[T] = new DenseTensor(other)
  def Tensor[@specialized(Float, Double) T: ClassTag](vector : BrzDenseVector[T])(implicit ev: TensorNumeric[T]) : Tensor[T] = torch.Tensor(torch.storage(vector.data), vector.offset + 1, Array(vector.length), Array(vector.stride))
  def Tensor(vector : DenseVector) : Tensor[Double] = torch.Tensor[Double](torch.storage(vector.toArray))
  def Tensor[@specialized(Float, Double) T: ClassTag](matrix : BrzDenseMatrix[T])(implicit ev: TensorNumeric[T]) : Tensor[T] = torch.Tensor(torch.storage(matrix.data), matrix.offset + 1, Array(matrix.rows, matrix.cols),
    if(matrix.isTranspose) Array(1, matrix.majorStride) else Array(matrix.majorStride, 1))
  def Tensor(matrix : DenseMatrix) : Tensor[Double] = {
    val strides = if(matrix.isTransposed) Array(matrix.numCols, 1) else Array(1, matrix.numRows)  // column major
    torch.Tensor(torch.storage(matrix.toArray), 1, Array(matrix.numRows, matrix.numCols), strides)
  }

  def Tensor[@specialized(Float, Double) T: ClassTag](indices : Array[Array[Int]], values : Storage[T], shape : Array[Int]) : Tensor[T] = {
    new SparseTensor[T](indices, values.asInstanceOf[Storage[T]], shape)
  }

  def Tensor[@specialized(Float, Double) T: ClassTag](rowIndices : Array[Int], columns: Array[Int],values : Storage[T], shape : Array[Int]) : Tensor[T] = {
    new SparseTensorCsr[T](rowIndices, columns, values.asInstanceOf[Storage[T]], shape)
  }

  def randperm[@specialized(Float, Double) T: ClassTag](size: Int)(implicit ev: TensorNumeric[T]) : Tensor[T] = DenseTensor.randperm[T](size)
  def expand[@specialized(Float, Double) T](tensor: Tensor[T], sizes: Int*) : Tensor[T] = tensor.expand(sizes.toArray)
  def expandAs[@specialized(Float, Double) T](tensor: Tensor[T], template: Tensor[T]) : Tensor[T] = tensor.expandAs(template)
  def repeatTensor[@specialized(Float, Double) T](tensor: Tensor[T], sizes: Int*) : Tensor[T] = tensor.repeatTensor(sizes.toArray)
  def storage[@specialized(Float, Double) T: ClassTag]() : Storage[T] = new ArrayStorage[T](new Array[T](0))
  def storage[@specialized(Float, Double) T: ClassTag](size : Int) : Storage[T] = new ArrayStorage[T](new Array[T](size))
  def storage[@specialized(Float, Double) T: ClassTag](data : Array[T]) : Storage[T] = new ArrayStorage[T](data)

  def load[T](fileName : String) : T = File.load[T](fileName)
  def loadObj[T](fileName : String) : T = File.loadObj[T](fileName)
  def save(data : Any, fileName : String,objectType : TorchObject) = File.save(data, fileName, objectType)
  def saveObj(obj : Serializable, fileName : String, isOverwrite : Boolean = false) = File.save(obj, fileName, isOverwrite)
}