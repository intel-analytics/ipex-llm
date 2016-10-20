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

package com.intel.analytics.sparkdl.utils

import scala.collection.mutable
import scala.collection.mutable.Map

/**
 * Simulate the Table data structure in lua
 *
 * @param state
 * @param topIndex
 */
class Table private[sparkdl](
  state: Map[Any, Any] = new mutable.HashMap[Any, Any](),
  // index of last element in the contiguous numeric number indexed elements start from 1
  private var topIndex: Int = 0
) extends Serializable with Activities {

  private[sparkdl] def this(data: Array[Any]) = {
    this(new mutable.HashMap[Any, Any](), 0)
    while (topIndex < data.length) {
      state.put(topIndex + 1, data(topIndex))
      topIndex += 1
    }
  }

  def getState(): Map[Any, Any] = this.state

  def get[T](key: Any): Option[T] = {
    if (!state.contains(key)) {
      return None
    }

    Option(state(key).asInstanceOf[T])
  }

  def apply[T](key: Any): T = {
    state(key).asInstanceOf[T]
  }

  def update(key: Any, value: Any): this.type = {
    state(key) = value
    if (key.isInstanceOf[Int] && topIndex + 1 == key.asInstanceOf[Int]) {
      topIndex += 1
      while (state.contains(topIndex + 1)) {
        topIndex += 1
      }
    }
    this
  }

  override def clone(): Table = {
    val result = new Table()

    for (k <- state.keys) {
      result(k) = state.get(k).get
    }

    result
  }

  override def toString(): String = {
    s"{${state.map { case (key: Any, value: Any) => s"$key: $value" }.mkString(", ")}}"
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[Table]) {
      return false
    }
    val other = obj.asInstanceOf[Table]
    if (this.eq(other)) {
      return true
    }
    if (this.state.keys.size != other.getState().keys.size) {
      return false
    }
    this.state.keys.map(key => {
      if (this.state(key) != other.getState()(key)) {
        return false
      }
    })
    return true
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()

    this.state.keys.map(key => {
      hash = hash * seed + key.hashCode()
      hash = hash * seed + this.state(key).hashCode()
    })

    hash
  }

  def remove[T](index: Int): Option[T] = {
    require(index > 0)

    if (topIndex >= index) {
      var i = index
      val result = state(index)
      while (i < topIndex) {
        state(i) = state(i + 1)
        i += 1
      }
      state.remove(topIndex)
      topIndex -= 1
      Some(result.asInstanceOf[T])
    } else if (state.contains(index)) {
      state.remove(index).asInstanceOf[Option[T]]
    } else {
      None
    }
  }

  def remove[T](): Option[T] = {
    if (topIndex != 0) {
      remove[T](topIndex)
    } else {
      None
    }
  }

  def insert[T](obj: T): this.type = update(topIndex + 1, obj)

  def insert[T](index: Int, obj: T): this.type = {
    require(index > 0)

    if (topIndex >= index) {
      var i = topIndex + 1
      topIndex += 1
      while (i > index) {
        state(i) = state(i - 1)
        i -= 1
      }
      update(index, obj)
    } else {
      update(index, obj)
    }

    this
  }

  def add(other: Table): this.type = {
    for (s <- other.getState().keys) {
      require(s.isInstanceOf[String])
      this.state(s) = other(s)
    }
    this
  }

  def length(): Int = state.size
}

object T {
  def apply(): Table = {
    new Table()
  }

  /**
   * Construct a table from a sequence of value.
   *
   * The index + 1 will be used as the key
   */
  def apply(data1: Any, datas: Any*): Table = {
    val firstElement = Array(data1)
    val otherElements = datas.toArray
    new Table(firstElement ++ otherElements)
  }

  /**
   * Construct a table from an array
   *
   * The index + 1 will be used as the key
   *
   * @param data
   * @return
   */
  def array(data: Array[Any]): Table = {
    new Table(data.toArray)
  }

  /**
   * Construct a table from a sequence of pair.
   */
  def apply(tuple: Tuple2[Any, Any], tuples: Tuple2[Any, Any]*): Table = {
    val table = new Table()
    table(tuple._1) = tuple._2
    for ((k, v) <- tuples) {
      table(k) = v
    }
    table
  }
}
