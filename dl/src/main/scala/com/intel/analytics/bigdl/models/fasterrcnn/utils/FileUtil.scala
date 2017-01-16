/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.models.fasterrcnn.utils

import java.io.File

import com.intel.analytics.bigdl.utils.{File => DlFile}

object FileUtil {
  def checkOrCreateDirs(outPath: String): Unit = {
    val f = new File(outPath)
    if (!f.exists()) {
      f.mkdirs()
    }
  }


  def existFile(f: String): Boolean = new java.io.File(f).exists()


  def load[M](filename: String): Option[M] = {
    try {
      if (existFile(filename)) return Some(DlFile.load[M](filename))
    } catch {
      case ex: Exception => None
    }
    None
  }
}
