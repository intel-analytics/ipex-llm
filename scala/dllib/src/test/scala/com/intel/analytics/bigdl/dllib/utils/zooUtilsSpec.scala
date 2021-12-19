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

package com.intel.analytics.bigdl.dllib.common

import org.apache.hadoop.fs.Path
import org.scalatest.{FlatSpec, Matchers}


class zooUtilsSpec extends FlatSpec with Matchers {
  val path: String = getClass.getClassLoader.getResource("qa").getPath
  val txtRelations: String = path + "/relations.txt"

  "Utils listFiles" should "work properly" in {
    val files = zooUtils.listPaths(path)
    assert(files.size == 3)
    val recursiveFiles = zooUtils.listPaths(path, true)
    assert(recursiveFiles.size == 13)
  }

  "Utils readBytes" should "work properly" in {
    val inputStream = zooUtils.open(txtRelations)
    val fileLen = inputStream.available()
    inputStream.close()
    val bytes = zooUtils.readBytes(txtRelations)
    assert(bytes.length == fileLen)
  }

  "Utils saveBytes" should "work properly" in {
    val fs = zooUtils.getFileSystem(path)
    // Generate random file
    val tmpFile = System.currentTimeMillis()
    val randomContent = new Array[Byte](1000)
    zooUtils.saveBytes(randomContent, path + "/" + tmpFile)
    // Delete random file
    fs.deleteOnExit(new Path(path + "/" + tmpFile))
    fs.close()
  }
}
