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

package com.intel.analytics.bigdl.dataset.text

import com.intel.analytics.bigdl.dataset.Sentence
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Represent a sentence
 *
 * @param _data
 * @param _label
 * @tparam T
 */
class LabeledSentence[T: ClassTag](
  protected var _data: Array[T],
  protected var _label: Array[T])
  (implicit ev: TensorNumeric[T])
  extends Sentence[T] {

  private var _dataLength: Int = if (_data == null) 0 else _data.length
  private var _labelLength: Int = if (_label == null) 0 else _label.length

  def this()(implicit ev: TensorNumeric[T]) = this(null, null)

  def this(dataLength: Int, labelLength: Int)
          (implicit ev: TensorNumeric[T]) = {
    this(new Array[T](dataLength),
      new Array[T](labelLength))
    _dataLength = dataLength
    _labelLength = labelLength
  }

  def copy(rawData: Array[T], rawLabel: Array[T]): this.type = {
    _dataLength = rawData.length
    _labelLength = rawLabel.length
    if (_data == null || _data.length <= _dataLength) {
      _data = new Array[T](_dataLength)
    }
    if (_label == null || _label.length != _labelLength) {
      _label = new Array[T](_labelLength)
    }
    ev.getType() match {
      case DoubleType =>
        Array.copy(rawData
          .asInstanceOf[Array[Double]], 0, _data
          .asInstanceOf[Array[Double]], 0, _dataLength)
        Array.copy(rawLabel
          .asInstanceOf[Array[Double]], 0, _label
          .asInstanceOf[Array[Double]], 0, _labelLength)
      case FloatType =>
        Array.copy(rawData
          .asInstanceOf[Array[Float]], 0, _data
          .asInstanceOf[Array[Float]], 0, _dataLength)
        Array.copy(rawLabel
          .asInstanceOf[Array[Float]], 0, _label
          .asInstanceOf[Array[Float]], 0, _labelLength)
      case t => throw new NotImplementedError(s"$t is not supported")
    }
    this
  }

  def getData(index: Int): T = {
    require(index >= 0 && index < _dataLength, "index out of boundary")
    _data(index)
  }

  def getLabel(index: Int): T = {
    require(index >= 0 && index < _labelLength, "index out of boundary")
    _label(index)
  }

  def copyToData(storage: Array[T], offset: Int): Unit = {
    require(_dataLength + offset <= storage.length)
    ev.getType() match {
      case DoubleType => Array.copy(_data
        .asInstanceOf[Array[Double]], 0, storage
        .asInstanceOf[Array[Double]], offset, _dataLength)
      case FloatType => Array.copy(_data
        .asInstanceOf[Array[Float]], 0, storage
        .asInstanceOf[Array[Float]], offset, _dataLength)
      case t => throw new NotImplementedError(s"$t is not supported")
    }
  }

  def copyToLabel(storage: Array[T], offset: Int): Unit = {
    require(_labelLength + offset <= storage.length)
    ev.getType() match {
      case DoubleType => Array.copy(_label
        .asInstanceOf[Array[Double]], 0, storage
        .asInstanceOf[Array[Double]], offset, _labelLength)
      case FloatType => Array.copy(_label
        .asInstanceOf[Array[Float]], 0, storage
        .asInstanceOf[Array[Float]], offset, _labelLength)
      case t => throw new NotImplementedError(s"$t is not supported")
    }
  }

  def copy(other: LabeledSentence[T]): LabeledSentence[T] = {
    this._labelLength = other._labelLength
    this._dataLength = other._dataLength
    if (this._data == null || this._data.length < this._dataLength) {
      this._data = new Array[T](this._dataLength)
    }
    if (this._label == null || this._label.length < this._labelLength) {
      this._label = new Array[T](this._labelLength)
    }
    ev.getType() match {
      case DoubleType => Array.copy(other._data
        .asInstanceOf[Array[Double]], 0, this._data
        .asInstanceOf[Array[Double]], 0, _dataLength)
        Array.copy(other._label
          .asInstanceOf[Array[Double]], 0, this._label
          .asInstanceOf[Array[Double]], 0, _labelLength)
      case FloatType => Array.copy(other._data
        .asInstanceOf[Array[Float]], 0, this._data
        .asInstanceOf[Array[Float]], 0, _dataLength)
        Array.copy(other._label
          .asInstanceOf[Array[Float]], 0, this._label
          .asInstanceOf[Array[Float]], 0, _labelLength)
      case t => throw new NotImplementedError(s"$t is not supported")
    }
    this
  }

  override def clone(): LabeledSentence[T] = {
    new LabeledSentence[T]().copy(this)
  }

  def labelLength(): Int = _labelLength

  def label(): Array[T] = _label

  override def data(): Array[T] = _data

  override def dataLength(): Int = _dataLength
}
