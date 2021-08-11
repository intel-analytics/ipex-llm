/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dataset


import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DenseType, SparseType, Storage, Tensor}
import org.apache.commons.lang3.SerializationUtils
import org.apache.zookeeper.KeeperException.UnimplementedException

import scala.reflect.ClassTag

/**
 * Class that represents the features and labels of a data sample.
 *
 * @tparam T numeric type
 */
abstract class Sample[T: ClassTag] extends Serializable {
  /**
   * First dimension length of index-th feature.
   * This function could be used to sort samples in [[DataSet]].
   *
   * @return
   */
  def featureLength(index: Int): Int

  /**
   * First dimension length of index-th label.
   * This function could be used to find the longest label.
   *
   * @return
   */
  def labelLength(index: Int): Int

  /**
   * Number of tensors in feature
   *
   * @return number of tensors in feature
   */
  def numFeature(): Int

  /**
   * Number of tensors in label
   *
   * @return number of tensors in label
   */
  def numLabel(): Int

  /**
   *@return A deep clone
   */
  override def clone(): this.type =
    SerializationUtils.clone(this)

  /**
   * Get feature tensor, for one feature Sample only.
   * You don't need to override this, because we have add
   * a default implement to throw exception.
   * @return feature tensor
   */
  def feature()(implicit ev: TensorNumeric[T]): Tensor[T]

  /**
   * Get feature tensor for given index
   * @param index index of specific sample
   */
  def feature(index: Int)(implicit ev: TensorNumeric[T]): Tensor[T]

  /**
   * Get label tensor, for one label Sample only.
   * You don't need to override this, because we have add
   * a default implement to throw exception.
   * @return label tensor
   */
  def label()(implicit ev: TensorNumeric[T]): Tensor[T]

  /**
   * Get label tensor for given index
   * @param index index of specific sample
   */
  def label(index: Int)(implicit ev: TensorNumeric[T]): Tensor[T]

  /**
   * Set data of feature and label.
   * @param featureData
   * @param labelData
   * @param featureSize
   * @param labelSize
   * @return
   */
  @deprecated("Old interface", "0.2.0")
  def set(
        featureData: Array[T],
        labelData: Array[T],
        featureSize: Array[Int],
        labelSize: Array[Int])(implicit ev: TensorNumeric[T]): Sample[T] = {
    throw new UnsupportedOperationException("Sample.set(): unimplemented deprecated method")
  }

  /**
   * Get feature sizes
   * @return feature sizes
   */
  def getFeatureSize(): Array[Array[Int]]


  /**
   * Get label sizes
   * @return label sizes
   */
  def getLabelSize(): Array[Array[Int]]

  /**
   * Get data
   * @return data
   */
  def getData(): Array[T]
}


/**
 * A kind of sample who use only one array
 */
class ArraySample[T: ClassTag] private[bigdl](
      private val data: Array[T],
      private val featureSize: Array[Array[Int]],
      private val labelSize: Array[Array[Int]]) extends Sample[T] {
  require(data != null, "Sample: Data couldn't be empty")
  require(featureSize != null, "Sample: Feature couldn't be empty")

  override def getData(): Array[T] = data

  override def featureLength(index: Int): Int = {
    require(null != featureSize, "featureSize is empty")
    featureSize(index)(0)
  }

  override def labelLength(index: Int): Int = {
    if (null != labelSize) {
      labelSize(index)(0)
    } else {
      0
    }
  }

  override def getFeatureSize(): Array[Array[Int]] = {
    featureSize
  }

  override def getLabelSize(): Array[Array[Int]] = {
    require(null != labelSize, "Sample doesn't have label")
    labelSize
  }

  override def numFeature(): Int = {
    require(null != featureSize, "featureSize is empty")
    featureSize.length
  }

  override def numLabel(): Int = {
    if (null == labelSize) {
      0
    } else {
      labelSize.length
    }
  }

  override def feature()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(this.numFeature == 1, "Only one Sample required in total" +
      s"got ${featureSize.length} feature Sample, please use feature(index) instead")
    feature(0)
  }

  override def feature(index: Int)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(this.numFeature > index, "feature index out of range")
    val featureOffSet = 1 + getFeatureSize().zipWithIndex.
      filter(_._2 < index).map(_._1.product).sum
    Tensor[T](Storage(data), featureOffSet, getFeatureSize()(index))
  }

  override def label(index: Int)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(this.numFeature > index, "label index out of range")
    if (this.numLabel > index) {
      val labelOffSet = 1 + getFeatureSize().map(_.product).sum + getLabelSize().zipWithIndex
        .filter(_._2 < index).map(_._1.product).sum
      Tensor[T](Storage[T](data), labelOffSet, labelSize(index))
    } else {
      null
    }
  }

  override def label()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(this.numLabel <= 1, "Only one Sample required in total " +
      s"got ${labelSize.length} label Sample, please use label(index) instead")
    label(0)
  }

  @deprecated("Old interface", "0.2.0")
  override def set(
           featureData: Array[T],
           labelData: Array[T],
           featureSize: Array[Int],
           labelSize: Array[Int])(implicit ev: TensorNumeric[T]): Sample[T] = {
    require(featureSize.sameElements(this.featureSize(0)) &&
      labelSize.sameElements(this.labelSize(0)), "size not match")

    ev.arraycopy(featureData, 0, data, 0, featureData.length)
    ev.arraycopy(labelData, 0, data, featureData.length, labelData.length)

    this
  }

  def canEqual(other: Any): Boolean = other.isInstanceOf[ArraySample[T]]

  override def equals(other: Any): Boolean = other match {
    case that: ArraySample[T] =>
      if (!(that canEqual this) ||
        !(data.deep == that.data.deep) ||
        !(featureSize.deep == that.featureSize.deep)) {
        return false
      }
      if (null != labelSize && null != that.labelSize) {
        labelSize.deep == that.labelSize.deep
      } else {
        null == labelSize & null == that.labelSize
      }
    case _ => false
  }

  override def hashCode(): Int = {
    val state = if (null == labelSize) Seq(data, featureSize) else Seq(data, featureSize, labelSize)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object ArraySample {
  private def typeCheck[T: ClassTag](tensor: Tensor[T]): Unit = {
    tensor.getTensorType match {
      case DenseType =>
        require(tensor.isContiguous(), s"tensor in ArraySample should be contiguous," +
          s" Please check your input.")
      case _ =>
        throw new IllegalArgumentException(s"ArraySample doesn't support ${tensor.getTensorType}")
    }
  }

  private def typeCheck[T: ClassTag](tensors: Array[Tensor[T]]): Unit = {
    tensors.foreach{tensor =>
      typeCheck(tensor)
    }
  }

  def apply[T: ClassTag](
        data: Array[T],
        featureSize: Array[Array[Int]],
        labelSize: Array[Array[Int]]): Sample[T] = {
    new ArraySample(data, featureSize, labelSize)
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensor)
    typeCheck(labelTensor)
    val data = new Array[T](featureTensor.nElement() + labelTensor.nElement())
    ev.arraycopy(featureTensor.storage().array(), featureTensor.storageOffset() - 1,
      data, 0, featureTensor.nElement())
    ev.arraycopy(labelTensor.storage().array(), labelTensor.storageOffset() - 1,
      data, featureTensor.nElement(), labelTensor.nElement())
    new ArraySample[T](data, getSize(featureTensor), getSize(labelTensor))
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        label: T)(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensor)
    val data = new Array[T](featureTensor.nElement() + 1)
    ev.arraycopy(featureTensor.storage().array(), featureTensor.storageOffset() - 1,
      data, 0, featureTensor.nElement())
    data(featureTensor.nElement()) = label
    new ArraySample[T](data, getSize(featureTensor), Array(Array(1)))
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensors)
    typeCheck(labelTensor)
    val tensors = featureTensors ++ Array(labelTensor)
    val data = new Array[T](tensors.map(_.nElement()).sum)
    copy(data, tensors)
    new ArraySample[T](data, getSize(featureTensors), getSize(labelTensor))
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]],
        labelTensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensors)
    typeCheck(labelTensors)
    val tensors = featureTensors ++ labelTensors
    val data = new Array[T](tensors.map(_.nElement()).sum)
    copy(data, tensors)
    new ArraySample[T](data, getSize(featureTensors), getSize(labelTensors))
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensor)
    val data = new Array[T](featureTensor.nElement())
    ev.arraycopy(featureTensor.storage().array(), featureTensor.storageOffset() - 1,
      data, 0, featureTensor.nElement())
    new ArraySample[T](data, getSize(featureTensor), null)
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensors)
    val data = new Array[T](featureTensors.map(_.nElement()).sum)
    copy(data, featureTensors)
    new ArraySample[T](data, getSize(featureTensors), null)
  }

  private def copy[T: ClassTag](
        data: Array[T],
        tensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Array[T] = {
    var offset = 0
    var i = 0
    while (i < tensors.length) {
      val tensor = tensors(i)
      require(tensor.isContiguous(), s"${i}-th tensor is not contiguous")
      ev.arraycopy(tensor.storage().array(), tensor.storageOffset() - 1,
        data, offset, tensor.nElement())
      offset += tensor.nElement()
      i += 1
    }
    data
  }

  private[bigdl] def getSize[T: ClassTag](tensors: Array[Tensor[T]]): Array[Array[Int]] = {
    tensors.map(_.size)
  }

  private[bigdl] def getSize[T: ClassTag](tensor: Tensor[T]): Array[Array[Int]] = {
    Array(tensor.size())
  }

  private[bigdl] def sameSize(a: Array[Array[Int]], b: Array[Array[Int]]): Boolean = {
    if (a.length != b.length) return false
    var i = 0
    while (i < a.length) {
      if (a(i).length != b(i).length) return false
      i += 1
    }
    true
  }
}

object Sample {
  def apply[T: ClassTag](
      data: Array[T],
      featureSize: Array[Array[Int]],
      labelSize: Array[Array[Int]]): Sample[T] = {
    ArraySample(data, featureSize, labelSize)
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    if (featureTensor.getTensorType == DenseType) {
      ArraySample(featureTensor, labelTensor)
    } else {
      TensorSample(featureTensor, labelTensor)
    }
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        label: T)(implicit ev: TensorNumeric[T]) : Sample[T] = {
    if (featureTensor.getTensorType == DenseType) {
      ArraySample(featureTensor, label)
    } else {
      TensorSample(featureTensor, label)
    }
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    if (featureTensors.exists(_.getTensorType == SparseType)) {
      TensorSample(featureTensors, labelTensor)
    } else {
      ArraySample(featureTensors, labelTensor)
    }
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]],
        labelTensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    if (featureTensors.exists(_.getTensorType == SparseType) ||
        labelTensors.exists(_.getTensorType == SparseType)) {
      TensorSample(featureTensors, labelTensors)
    } else {
      ArraySample(featureTensors, labelTensors)
    }
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    if (featureTensor.getTensorType == SparseType) {
      TensorSample(featureTensor)
    } else {
      ArraySample(featureTensor)
    }
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    if (featureTensors.exists(_.getTensorType == SparseType)) {
      TensorSample(featureTensors)
    } else {
      ArraySample(featureTensors)
    }
  }
}

/**
 * A kind of Sample who hold both DenseTensor and SparseTensor as features.
 * @param features feature tensors
 * @param labels label tensors
 * @tparam T numeric type
 */
class TensorSample[T: ClassTag] private[bigdl] (
      val features: Array[Tensor[T]],
      val labels: Array[Tensor[T]]) extends Sample[T] {
  protected val featureSize = features.map(_.size())
  protected val labelSize = labels.map(_.size())

  def featureLength(index: Int): Int = {
    features(0).size(1)
  }

  def labelLength(index: Int): Int = {
    labels(0).size(1)
  }

  def numFeature(): Int = {
    features.length
  }

  def numLabel(): Int = {
    labels.length
  }

  def getFeatureSize(): Array[Array[Int]] = {
    featureSize
  }

  def getLabelSize(): Array[Array[Int]] = {
    labelSize
  }

  def getData(): Array[T] = {
    throw new UnimplementedException()
  }

  override def feature()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(this.numFeature == 1, "only sample with one feature supported")
    this.feature(0)
  }

  override def feature(index: Int)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(index < this.numFeature, "Index out of range")
    this.features(index)
  }

  override def label()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(this.numLabel <= 1, "only sample with at most one label supported")
    if (this.numLabel == 1) this.label(0) else null
  }

  override def label(index: Int)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(index < this.numFeature, "Index out of range")
    if (index < this.numLabel) this.labels(index) else null
  }

}

object TensorSample {
  private def typeCheck[T: ClassTag](tensor: Tensor[T]): Unit = {
    tensor.getTensorType match {
      case DenseType =>
        require(tensor.isContiguous(), s"tensor in TensorSample should be contiguous," +
          s" Please check your input.")
      case SparseType =>
      case _ =>
        throw new IllegalArgumentException(s"TensorSample doesn't support ${tensor.getTensorType}")
    }
  }

  private def typeCheck[T: ClassTag](tensors: Array[Tensor[T]]): Unit = {
    tensors.foreach{tensor =>
      typeCheck(tensor)
    }
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensors)
    new TensorSample[T](featureTensors, Array())
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensor)
    new TensorSample[T](Array(featureTensor), Array())
  }
  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]],
        labelTensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensors)
    typeCheck(labelTensors)
    new TensorSample[T](featureTensors, labelTensors)
  }

  def apply[T: ClassTag](
        featureTensors: Array[Tensor[T]],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensors)
    typeCheck(labelTensor)
    new TensorSample[T](featureTensors, Array(labelTensor))
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensor)
    typeCheck(labelTensor)
    new TensorSample[T](Array(featureTensor), Array(labelTensor))
  }

  def apply[T: ClassTag](
        featureTensor: Tensor[T],
        label: T)(implicit ev: TensorNumeric[T]) : Sample[T] = {
    typeCheck(featureTensor)
    new TensorSample[T](Array(featureTensor), Array(Tensor(1).fill(label)))
  }

  /**
   * Create a TensorSample which is able to contains Tensors with different types.
   *
   * @tparam T main type
   * @param featureTensors feature tensors
   * @param labelTensors label tensors, can be null or empty, default value is null
   * @return TensorSample
   */
  def create[T: ClassTag](
      featureTensors: Array[Tensor[_]],
      labelTensors: Array[Tensor[_]] = null)
    (implicit ev: TensorNumeric[T]) : Sample[T] = {
    if (labelTensors == null || labelTensors.isEmpty) {
      TensorSample(wrapType(featureTensors))
    } else {
      TensorSample(wrapType(featureTensors), wrapType(labelTensors))
    }
  }

  private def wrapType[T: ClassTag](tensor: Array[Tensor[_]])
    (implicit ev: TensorNumeric[T]): Array[Tensor[T]] = {
    tensor.map(_.asInstanceOf[Tensor[T]])
  }

}
