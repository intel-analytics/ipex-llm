package com.intel.analytics.dllib.lib.tensor

import scala.collection.mutable
import scala.collection.mutable.Map

/**
 * Simulate the Table data structure in lua
 *
 * @param state
 * @param topIndex
 */
class Table private[tensor] (
    state : Map[Any, Any] = new mutable.HashMap[Any, Any](),
    private var topIndex : Int = 0  // index of last element in the contiguous numeric number indexed elements start from 1
  ) extends Serializable {

  private[tensor] def this(data : Array[Any]) = {
    this(new mutable.HashMap[Any, Any](), 0)
    while(topIndex < data.length) {
      state.put(topIndex + 1, data(topIndex))
      topIndex += 1
    }
  }

  def getState() : Map[Any, Any] = this.state

  def get[T](key: Any): Option[T] = {
    if (!state.contains(key)) {
      return None
    }

    Option(state(key).asInstanceOf[T])
  }

  def apply[T](key : Any) : T = {
    state(key).asInstanceOf[T]
  }

  def update(key: Any, value: Any): this.type = {
    state(key) = value
    if(key.isInstanceOf[Int] && topIndex + 1== key.asInstanceOf[Int]) {
      topIndex += 1
      while(state.contains(topIndex + 1)) {
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

  def remove[T](index: Int) : Option[T] = {
    require(index > 0)

    if(topIndex >= index) {
      var i = index
      val result = state(index)
      while(i < topIndex) {
        state(i) = state(i + 1)
        i += 1
      }
      state.remove(topIndex)
      topIndex -= 1
      Some(result.asInstanceOf[T])
    } else if(state.contains(index)) {
      state.remove(index).asInstanceOf[Option[T]]
    } else {
      None
    }
  }

  def remove[T]() : Option[T] = {
    if(topIndex != 0)
      remove[T](topIndex)
    else
      None
  }

  def insert[T](obj: T): this.type = update(topIndex + 1, obj)

  def insert[T](index: Int, obj: T): this.type = {
    require(index > 0)

    if(topIndex >= index) {
      var i = topIndex + 1
      topIndex += 1
      while(i > index) {
        state(i) = state(i - 1)
        i -= 1
      }
      update(index, obj)
    } else {
      update(index, obj)
    }

    this
  }

  def add(other : Table) : this.type = {
    for(s <- other.getState().keys) {
      require(s.isInstanceOf[String])
      this.state(s) = other(s)
    }
    this
  }

  def length() : Int = state.size
}

object T {
  def apply() : Table = {
    new Table()
  }

  /**
   * Construct a table from a sequence of value.
   *
   * The index + 1 will be used as the key
   */
  def apply(data1: Any, datas : Any*) : Table = {
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
  def array (data : Array[Any]) : Table = {
    new Table(data.toArray)
  }

  /**
   * Construct a table from a sequence of pair.
   */
  def apply(tuple : Tuple2[Any, Any], tuples : Tuple2[Any, Any]*) : Table = {
    val table = new Table()
    table(tuple._1) = tuple._2
    for((k, v) <- tuples) {
      table(k) = v
    }
    table
  }
}
