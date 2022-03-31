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

package com.intel.analytics.bigdl.dllib.tensor

/**
 * This package is used to provide concrete implementations of the conversions
 * between numeric primitives. The idea here is that the Numeric trait can
 * extend these traits to inherit the conversions.
 *
 * We can also use these implementations to provide a way to convert from
 * A -> B, where both A and B are generic Numeric types. Without a separate
 * trait, we'd have circular type definitions when compiling Numeric.
 */

import com.intel.analytics.bigdl.dllib.utils.Log4Error

import scala.language.implicitConversions
import scala.{specialized => spec}

/**
 * Conversions to type.
 *
 * An object implementing ConvertableTo[A] provides methods to go
 * from number types to A.
 */
trait ConvertableTo[@spec A] {
  implicit def fromFloat(a: Float): A

  implicit def fromDouble(a: Double): A

  implicit def fromInt(a: Int): A

  implicit def fromShort(a: Short): A

  implicit def fromLong(a: Long): A

  implicit def fromBoolean(a: Boolean): A
}

trait ConvertableToLong extends ConvertableTo[Long] {
  implicit def fromFloat(a: Float): Long = a.toLong

  implicit def fromDouble(a: Double): Long = a.toLong

  implicit def fromInt(a: Int): Long = a.toLong

  implicit def fromShort(a: Short): Long = a.toLong

  implicit def fromLong(a: Long): Long = a.toLong

  implicit def fromBoolean(a: Boolean): Long = if (a) 1L else 0L
}


trait ConvertableToShort extends ConvertableTo[Short] {
  implicit def fromFloat(a: Float): Short = a.toShort

  implicit def fromDouble(a: Double): Short = a.toShort

  implicit def fromInt(a: Int): Short = a.toShort

  implicit def fromShort(a: Short): Short = a.toShort

  implicit def fromLong(a: Long): Short = a.toShort

  implicit def fromBoolean(a: Boolean): Short = if (a) 1 else 0
}


trait ConvertableToFloat extends ConvertableTo[Float] {
  implicit def fromFloat(a: Float): Float = a

  implicit def fromDouble(a: Double): Float = a.toFloat

  implicit def fromInt(a: Int): Float = a.toFloat

  implicit def fromShort(a: Short): Float = a.toFloat

  implicit def fromLong(a: Long): Float = a.toFloat

  implicit def fromBoolean(a: Boolean): Float = if (a) 1.0f else 0.0f
}

trait ConvertableToDouble extends ConvertableTo[Double] {
  implicit def fromFloat(a: Float): Double = a.toDouble

  implicit def fromDouble(a: Double): Double = a

  implicit def fromInt(a: Int): Double = a.toDouble

  implicit def fromShort(a: Short): Double = a.toDouble

  implicit def fromLong(a: Long): Double = a.toDouble

  implicit def fromBoolean(a: Boolean): Double = if (a) 1.0 else 0.0
}

trait ConvertableToInt extends ConvertableTo[Int] {
  implicit def fromFloat(a: Float): Int = a.toInt

  implicit def fromDouble(a: Double): Int = a.toInt

  implicit def fromInt(a: Int): Int = a

  implicit def fromShort(a: Short): Int = a.toShort

  implicit def fromLong(a: Long): Int = a.toInt

  implicit def fromBoolean(a: Boolean): Int = if (a) 1 else 0
}

object ConvertableTo {

  implicit object ConvertableToFloat extends ConvertableToFloat

  implicit object ConvertableToDouble extends ConvertableToDouble

  implicit object ConvertableToInt extends ConvertableToInt

  implicit object ConvertableToShort extends ConvertableToShort

  implicit object ConvertableToLong extends ConvertableToLong
}


/**
 * Conversions from type.
 *
 * An object implementing ConvertableFrom[A] provides methods to go
 * from A to number types (and String).
 */
trait ConvertableFrom[@spec A] {
  implicit def toFloat(a: A): Float

  implicit def toDouble(a: A): Double

  implicit def toShort(a: A): Short

  implicit def toLong(a: A): Long

  implicit def toInt(a: A): Int

  implicit def toString(a: A): String

  implicit def toChar(a: A): Char

  implicit def toByte(a: A): Byte

  implicit def toBoolean(a: A): Boolean
}

trait ConvertableFromFloat extends ConvertableFrom[Float] {
  implicit def toFloat(a: Float): Float = a

  implicit def toDouble(a: Float): Double = a.toDouble

  implicit def toInt(a: Float): Int = a.toInt

  implicit def toShort(a: Float): Short = a.toShort

  implicit def toLong(a: Float): Long = a.toLong

  implicit def toString(a: Float): String = a.toString

  implicit def toChar(a: Float): Char = a.toChar

  implicit def toByte(a: Float): Byte = a.toByte

  implicit def toBoolean(a: Float): Boolean = {
    Log4Error.invalidOperationError(false, "Float cannot be cast to Boolean type")
    false
  }
}

trait ConvertableFromDouble extends ConvertableFrom[Double] {
  implicit def toFloat(a: Double): Float = a.toFloat

  implicit def toDouble(a: Double): Double = a

  implicit def toInt(a: Double): Int = a.toInt

  implicit def toShort(a: Double): Short = a.toShort

  implicit def toLong(a: Double): Long = a.toLong

  implicit def toString(a: Double): String = a.toString

  implicit def toChar(a: Double): Char = a.toChar

  implicit def toByte(a: Double): Byte = a.toByte

  implicit def toBoolean(a: Double): Boolean = {
    Log4Error.invalidOperationError(false, "Float cannot be cast to Boolean type")
    false
  }
}

trait ConvertableFromInt extends ConvertableFrom[Int] {
  implicit def toFloat(a: Int): Float = a.toFloat

  implicit def toDouble(a: Int): Double = a.toDouble

  implicit def toInt(a: Int): Int = a

  implicit def toShort(a: Int): Short = a.toShort

  implicit def toLong(a: Int): Long = a.toLong

  implicit def toString(a: Int): String = a.toString

  implicit def toChar(a: Int): Char = a.toChar

  implicit def toByte(a: Int): Byte = a.toByte

  implicit def toBoolean(a: Int): Boolean = {
    Log4Error.invalidOperationError(false, "Float cannot be cast to Boolean type")
    false
  }
}

trait ConvertableFromShort extends ConvertableFrom[Short] {
  implicit def toFloat(a: Short): Float = a.toFloat

  implicit def toDouble(a: Short): Double = a.toDouble

  implicit def toInt(a: Short): Int = a

  implicit def toShort(a: Short): Short = a.toShort

  implicit def toLong(a: Short): Long = a.toLong

  implicit def toString(a: Short): String = a.toString

  implicit def toChar(a: Short): Char = a.toChar

  implicit def toByte(a: Short): Byte = a.toByte

  implicit def toBoolean(a: Short): Boolean = {
    Log4Error.invalidOperationError(false, "Float cannot be cast to Boolean type")
    false
  }
}

trait ConvertableFromLong extends ConvertableFrom[Long] {
  implicit def toFloat(a: Long): Float = a.toFloat

  implicit def toDouble(a: Long): Double = a.toDouble

  implicit def toInt(a: Long): Int = a

  implicit def toShort(a: Long): Short = a.toShort

  implicit def toLong(a: Long): Long = a.toLong

  implicit def toString(a: Long): String = a.toString

  implicit def toChar(a: Long): Char = a.toChar

  implicit def toByte(a: Long): Byte = a.toByte

  implicit def toBoolean(a: Long): Boolean = {
    Log4Error.invalidOperationError(false, "Float cannot be cast to Boolean type")
    false
  }
}

trait ConvertableFromBoolean extends ConvertableFrom[Boolean] {
  implicit def toFloat(a: Boolean): Float = {
    Log4Error.invalidOperationError(false, "Boolean cannot be cast to Float type")
    0.0f
  }

  implicit def toDouble(a: Boolean): Double = {
    Log4Error.invalidOperationError(false, "Boolean cannot be cast to Double type")
    0.0
  }

  implicit def toInt(a: Boolean): Int = {
    Log4Error.invalidOperationError(false, "Boolean cannot be cast to Int type")
    0
  }

  implicit def toShort(a: Boolean): Short = {
    Log4Error.invalidOperationError(false, "Boolean cannot be cast to Short type")
    0
  }

  implicit def toLong(a: Boolean): Long = {
    Log4Error.invalidOperationError(false, "Boolean cannot be cast to Long type")
    0
  }

  implicit def toString(a: Boolean): String = a.toString

  implicit def toChar(a: Boolean): Char = {
    Log4Error.invalidOperationError(false, "Boolean cannot be cast to char type")
    'a'
  }

  implicit def toByte(a: Boolean): Byte = {
    Log4Error.invalidOperationError(false, "Boolean cannot be cast to byte type")
    0
  }

  implicit def toBoolean(a: Boolean): Boolean = a
}

trait ConvertableFromString extends ConvertableFrom[String] {
  implicit def toFloat(a: String): Float = {
    Log4Error.invalidOperationError(false, "string cannot be cast to Float type")
    0.0f
  }

  implicit def toDouble(a: String): Double = {
    Log4Error.invalidOperationError(false, "string cannot be cast to Double type")
    0.0
  }

  implicit def toInt(a: String): Int = {
    Log4Error.invalidOperationError(false, "string cannot be cast to Int type")
    0
  }

  implicit def toShort(a: String): Short = {
    Log4Error.invalidOperationError(false, "string cannot be cast to Short type")
    0
  }

  implicit def toLong(a: String): Long = {
    Log4Error.invalidOperationError(false, "string cannot be cast to Long type")
    0
  }

  implicit def toChar(a: String): Char = {
    Log4Error.invalidOperationError(false, "string cannot be cast to char type")
    '0'
  }

  implicit def toBoolean(a: String): Boolean = {
    Log4Error.invalidOperationError(false, "string cannot be cast to Boolean type")
    false
  }

  implicit def toByte(a: String): Byte = {
    Log4Error.invalidOperationError(false, "string cannot be cast to byte type")
    0
  }

  implicit def toString(a: String): String = a
}

trait ConvertableFromChar extends ConvertableFrom[Char] {
  implicit def toFloat(a: Char): Float = a.toFloat

  implicit def toDouble(a: Char): Double = a.toDouble

  implicit def toInt(a: Char): Int = a.toInt

  implicit def toShort(a: Char): Short = a.toShort

  implicit def toLong(a: Char): Long = a.toLong

  implicit def toBoolean(a: Char): Boolean = {
    Log4Error.invalidOperationError(false, "string cannot be cast to boolean type")
    false
  }

  implicit def toString(a: Char): String = a.toString

  implicit def toChar(a: Char): Char = a

  implicit def toByte(a: Char): Byte = a.toByte

}

trait ConvertableFromByte extends ConvertableFrom[Byte] {
  implicit def toFloat(a: Byte): Float = a.toFloat

  implicit def toDouble(a: Byte): Double = a.toDouble

  implicit def toInt(a: Byte): Int = a.toInt

  implicit def toShort(a: Byte): Short = a.toShort

  implicit def toLong(a: Byte): Long = a.toLong

  implicit def toBoolean(a: Byte): Boolean = {
    Log4Error.invalidOperationError(false, "byte cannot be cast to boolean type")
    false
  }

  implicit def toString(a: Byte): String = a.toString

  implicit def toByte(a: Byte): Byte = a

  implicit def toChar(a: Byte): Char = a.toChar
}

object ConvertableFrom {

  implicit object ConvertableFromFloat extends ConvertableFromFloat

  implicit object ConvertableFromDouble extends ConvertableFromDouble

  implicit object ConvertableFromInt extends ConvertableFromInt

  implicit object ConvertableFromChar extends ConvertableFromChar

  implicit object ConvertableFromShort extends ConvertableFromShort

  implicit object ConvertableFromLong extends ConvertableFromLong

  implicit object ConvertableFromString extends ConvertableFromString

  implicit object ConvertableFromBoolean extends ConvertableFromBoolean

  implicit object ConvertableFromByte extends ConvertableFromByte
}

