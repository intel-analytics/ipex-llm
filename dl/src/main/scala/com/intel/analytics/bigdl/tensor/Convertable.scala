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

package com.intel.analytics.bigdl.tensor

/**
 * This package is used to provide concrete implementations of the conversions
 * between numeric primitives. The idea here is that the Numeric trait can
 * extend these traits to inherit the conversions.
 *
 * We can also use these implementations to provide a way to convert from
 * A -> B, where both A and B are generic Numeric types. Without a separate
 * trait, we'd have circular type definitions when compiling Numeric.
 */

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
}

trait ConvertableToFloat extends ConvertableTo[Float] {
  implicit def fromFloat(a: Float): Float = a

  implicit def fromDouble(a: Double): Float = a.toFloat

  implicit def fromInt(a: Int): Float = a.toFloat
}

trait ConvertableToDouble extends ConvertableTo[Double] {
  implicit def fromFloat(a: Float): Double = a.toDouble

  implicit def fromDouble(a: Double): Double = a

  implicit def fromInt(a: Int): Double = a.toDouble
}

trait ConvertableToInt extends ConvertableTo[Int] {
  implicit def fromFloat(a: Float): Int = a.toInt

  implicit def fromDouble(a: Double): Int = a.toInt

  implicit def fromInt(a: Int): Int = a
}

object ConvertableTo {

  implicit object ConvertableToFloat extends ConvertableToFloat

  implicit object ConvertableToDouble extends ConvertableToDouble

  implicit object ConvertableToInt extends ConvertableToInt

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

  implicit def toInt(a: A): Int
}

trait ConvertableFromFloat extends ConvertableFrom[Float] {
  implicit def toFloat(a: Float): Float = a

  implicit def toDouble(a: Float): Double = a.toDouble

  implicit def toInt(a: Float): Int = a.toInt
}

trait ConvertableFromDouble extends ConvertableFrom[Double] {
  implicit def toFloat(a: Double): Float = a.toFloat

  implicit def toDouble(a: Double): Double = a

  implicit def toInt(a: Double): Int = a.toInt
}

trait ConvertableFromInt extends ConvertableFrom[Int] {
  implicit def toFloat(a: Int): Float = a.toFloat

  implicit def toDouble(a: Int): Double = a.toDouble

  implicit def toInt(a: Int): Int = a
}

object ConvertableFrom {

  implicit object ConvertableFromFloat extends ConvertableFromFloat

  implicit object ConvertableFromDouble extends ConvertableFromDouble

  implicit object ConvertableFromInt extends ConvertableFromInt

}

