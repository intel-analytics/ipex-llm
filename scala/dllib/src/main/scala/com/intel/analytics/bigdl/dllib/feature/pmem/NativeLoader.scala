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

package com.intel.analytics.zoo.feature.pmem

import java.io.{File, FileOutputStream}
import java.nio.channels.{Channels, FileChannel, ReadableByteChannel}
import java.nio.file.{Files, Path}

import scala.collection.mutable.ArrayBuffer

class NativeLoader {
  val librariesInJar = new ArrayBuffer[String]()
  librariesInJar.append("persistent_memory_allocator")

  def init(): Unit = {
    val tempDir = Files.createTempDirectory("persistent.memory.")

    copyFromJarToTmpDir(tempDir)

    librariesInJar.foreach(loadLibrary(_, tempDir))
  }

  private def libraryName(name: String): String = {
    return "lib" + name + ".so"
  }

  def copyFromJarToTmpDir(tempDir: Path): Unit = {
    librariesInJar.foreach {name =>
      val library = libraryName(name)
      val src = resource(library)
      copyLibraryToTemp(src, library, tempDir);
      src.close()
    }
  }

  private def resource(name: String): ReadableByteChannel = {
    val relativePath = "/native/" + name
    val url = classOf[NativeLoader].getResource(relativePath)
    if (url == null) {
      throw new Error(
        "Can't find the library " + name + s" in the resource folder: ${relativePath}")
    }

    val in = classOf[NativeLoader].getResourceAsStream(relativePath)
    val src = Channels.newChannel(in);
    src
  }

  private def copyLibraryToTemp(src: ReadableByteChannel, name: String,
    tempDir: Path): Unit = {
    val tempFile = new File(tempDir.toFile() + File.separator + name);

    var dst: FileChannel = null
    try {
      dst = new FileOutputStream(tempFile).getChannel();
      dst.transferFrom(src, 0, Long.MaxValue);
    } finally {
      if (dst != null) {
        dst.close()
      }
    }
  }

  private def deleteAll(tempDir: Path) {
    val dir = tempDir.toFile()
    dir.listFiles().foreach { f =>
      f.delete()
    }

    dir.delete()
  }

  private def loadLibrary(name: String, tempDir: Path) {
    System.load(tempDir.toString() + File.separator + libraryName(name))
  }
}
