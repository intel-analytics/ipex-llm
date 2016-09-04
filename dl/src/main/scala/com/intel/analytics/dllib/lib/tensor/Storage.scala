package com.intel.analytics.dllib.lib.tensor

import com.intel.analytics.dllib.lib.tensor.TensorType.DataType

/**
 * Storage defines a simple storage interface that controls the underlying storage for any tensor object.
 *
 */
trait Storage[T] extends Iterable[T] with Serializable {

  /**
   * Returns the number of elements in the storage. The original method name in torch is size, which is conflict with
   * Iterable
   *
   * @return
   */
  def length() : Int

  /**
   * Returns the element at position index in the storage. Valid range of index is 0 to length() -1
   *
   * @param index
   * @return
   */
  //ToDo: make the index range from 1 to length() like torch
  def apply(index : Int) : T

  /**
   * Set the element at position index in the storage. Valid range of index is 1 to length()
   *
   * @param index
   * @param value
   */
  def update(index : Int, value : T) : Unit

  /**
   * Copy another storage. The types of the two storages might be different: in that case a conversion of types occur
   *  (which might result, of course, in loss of precision or rounding). This method returns itself.
   *
   * @param source
   * @return
   */
  def copy(source : Storage[T], offset : Int, sourceOffset : Int, length : Int) : this.type
  def copy(source : Storage[T]) : this.type = copy(source, 0, 0, length())

  /**
   * Fill the Storage with the given value. This method returns itself.
   *
   * @param value
   * @param offset offset start from 1
   * @param length length of fill part
   * @return
   */
  def fill(value : T, offset : Int, length : Int) : this.type

  /**
   * Resize the storage to the provided size. The new contents are undetermined. This function returns itself
   * @param size
   * @return
   */
  def resize(size : Long) : this.type

  /**
   * Convert the storage to an array
   * @return
   */
  def array() : Array[T]

  /**
   * Get the element type in the storage
   * @return
   */
//  def getType() : DataType

  /**
    * Share the same underlying storage
    *
    * @param other
    * @return
    */
  def set(other : Storage[T]) : this.type
}
