/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.common

import java.io.{InputStream, ObjectInputStream, ObjectStreamClass}

final class CheckedObjectInputStream(checkedClass: Class[_], inputStream: InputStream)
  extends ObjectInputStream(inputStream) {

  private var firstCall: Boolean = true

  @throws[ClassNotFoundException]("if no associated class can be found")
  @throws[UnsupportedOperationException]("if the identified class is not valid " +
    "for the declared serialization classes")
  override protected def resolveClass(objectStreamClass: ObjectStreamClass): Class[_] = {
    if (firstCall) {
      firstCall = false
      val resolvedClass = super.resolveClass(objectStreamClass)
      if (checkedClass.isAssignableFrom(resolvedClass)) {
        resolvedClass
      } else {
        throw new UnsupportedOperationException(s"Illegal serialization of class:" +
          s" $resolvedClass. Supported objects must be instances of ${checkedClass.getName}")
      }
    } else {
      super.resolveClass(objectStreamClass)
    }
  }
}
