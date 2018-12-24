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

package com.intel.analytics.bigdl.nn.mkldnn

import java.io.File
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.mkldnn.models.Vgg_16
import org.scalatest.{FlatSpec, Matchers}

class SerializeModelSpec extends FlatSpec with Matchers {

  "Save a model" should "work correctly" in {
    val identity = System.identityHashCode(this).toString
    val name = "vgg_16." + identity
    val tmpdir = System.getProperty("java.io.tmpdir")
    val path = Paths.get(tmpdir, name).toAbsolutePath

    val model = Vgg_16(32, 1000)
    println(s"generate the model file ${path.toString}")
    model.save(path.toString, true)
    val loaded = Module.load[Float](path.toString)

    println(s"delete the model file ${path.toString}")
    Files.deleteIfExists(path)
  }

}
