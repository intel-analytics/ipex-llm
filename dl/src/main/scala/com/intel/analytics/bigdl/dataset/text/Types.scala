/*
 * Licensed to the Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.dataset.text

import com.intel.analytics.bigdl.dataset.Sentence

 /**
  * Represent a sentence
  *
  * @param _data
  * @param _label
  */
class LabeledSentence[T](
    protected var _data: Array[T],
    protected var _label: Array[T]) extends Sentence {

   private var _dataLength: Int = 0
   private var _labelLength: Int = 0

  def this() = this(new Array[T](0), new Array[T](0))

  def this(dataLength: Int, labelLength: Int) = {
    this(new Array[T](dataLength), new Array[T](labelLength))
    _dataLength = dataLength
    _labelLength = labelLength
  }

  def copy(rawData: Array[T], rawLabel: Array[T]): this.type = {
    _dataLength = rawData.length
    _labelLength = rawLabel.length
    if (_data.length < _dataLength) {
      _data = new Array[T](_dataLength)
    }
    rawData.copyToArray(_data)
    if (_label.length < _labelLength) {
      _label = new Array[T](_labelLength)
    }
    rawLabel.copyToArray(_label)
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
    val frameLength = _dataLength
    require(frameLength + offset <= storage.length)
    _data.copyToArray(storage, offset)
  }

  def copyToLabel(storage: Array[T], offset: Int): Unit = {
    val frameLength = _labelLength
    require(frameLength + offset <= storage.length)
    _label.copyToArray(storage, offset)
  }

  def copy(other: LabeledSentence[T]): LabeledSentence[T] = {
    this._labelLength = other._labelLength
    this._dataLength = other._dataLength
    if (this._data.length < this._dataLength) {
      this._data = new Array[T](this._dataLength)
    }
    if (this._label.length < this._labelLength) {
      this._label = new Array[T](this._labelLength)
    }
    other._data.copyToArray(this._data)
    other._label.copyToArray(this._label)
    this
  }

  override def clone(): LabeledSentence[T] = {
    new LabeledSentence[T]().copy(this)
  }

  def labelLength(): Int = _labelLength

  def label(): Array[T] = _label

  override def content(): Array[T] = _data

  override def length(): Int = _dataLength
}
