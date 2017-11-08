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
package com.intel.analytics.bigdl.utils.tf

import org.scalatest.{FlatSpec, Matchers}
import java.io.{File => JFile}


class TFRecordIteratorSpec extends FlatSpec with Matchers {

  "TFRecordIterator " should "be able to read .tfrecord file" in {
    val resource = getClass.getClassLoader.getResource("tf")
    val path = processPath(resource.getPath) + JFile.separator + "text.tfrecord"
    val file = new JFile(path)

    val iter = TFRecordIterator(file)

    iter.map(a => new String(a)).toSeq should be (Seq("abcd"))
  }

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

}
