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

package com.intel.analytics.bigdl.serving.pipeline

import java.io.{BufferedReader, File, IOException, InputStreamReader}
import java.util
import java.util.concurrent.Executors

import org.apache.commons.io.IOUtils
import redis.embedded.{Redis, RedisExecProvider, RedisServerBuilder}
import redis.embedded.exceptions.EmbeddedRedisException

import scala.util.control.Breaks.{break, breakable}

// The RedisServer implementation is based on:
//    https://github.com/kstyrc/embedded-redis
class PrintReaderRunnable(val reader: BufferedReader) extends Runnable {
  override def run(): Unit = {
    try
      this.readLines()
    finally IOUtils.closeQuietly(this.reader)
  }

  def readLines(): Unit = {
    breakable{
      while ( {
        true
      }) {
        try {
          var line = this.reader.readLine()
          if (line != null) {
            System.out.println(line)
            break()
          }
        } catch {
          case var2: IOException =>
            var2.printStackTrace()
        }
        return
      }
    }
  }
}

abstract class AbstractRedisInstance protected(val port: Int) extends Redis {
  protected var args: util.List[String] = _
  private var active = false
  private var redisProcess: Process = _
  final private val executor = Executors.newSingleThreadExecutor

  override def isActive: Boolean = this.active

  @throws[EmbeddedRedisException]
  override def start(): Unit = {
    if (this.active) throw new EmbeddedRedisException(
      "This redis server instance is already running...")
    else try {
      this.redisProcess = this.createRedisProcessBuilder.start
      this.logErrors()
      this.awaitRedisServerReady()
      this.active = true
    } catch {
      case var2: IOException =>
        throw new EmbeddedRedisException("Failed to start Redis instance", var2)
    }
  }

  private def logErrors(): Unit = {
    val errorStream = this.redisProcess.getErrorStream
    val reader = new BufferedReader(new InputStreamReader(errorStream))
    val printReaderTask = new PrintReaderRunnable(reader)
    this.executor.submit(printReaderTask)
  }

  @throws[IOException]
  private def awaitRedisServerReady(): Unit = {
    val reader = new BufferedReader(new InputStreamReader(this.redisProcess.getInputStream))
    var outputLine = ""
    try
        do {
          outputLine = reader.readLine
          if (outputLine == null) throw new RuntimeException(
            "Can't start redis server. Check logs for details.")
        } while ( {
          !outputLine.matches(this.redisReadyPattern)
        })
    finally IOUtils.closeQuietly(reader)
  }

  protected def redisReadyPattern: String

  private def createRedisProcessBuilder = {
    val executable = new File(this.args.get(0).asInstanceOf[String])
    val pb = new ProcessBuilder(this.args)
    pb.directory(executable.getParentFile)
    pb
  }

  @throws[EmbeddedRedisException]
  override def stop(): Unit = {
    if (this.active) {
      this.redisProcess.destroy()
      this.tryWaitFor()
      this.active = false
    }
  }

  private def tryWaitFor(): Unit = {
    try
      this.redisProcess.waitFor
    catch {
      case var2: InterruptedException =>
        throw new EmbeddedRedisException("Failed to stop redis instance", var2)
    }
  }

  override def ports: util.List[Integer] = util.Arrays.asList(this.port)
}


object RedisServer {
  private val REDIS_READY_PATTERN = ".*The server is now ready to accept connections on port.*"
  private val DEFAULT_REDIS_PORT = 6379

  def builder : RedisServerBuilder = new RedisServerBuilder
}

class RedisEmbeddedReImpl @throws[IOException]
(port : Int) extends AbstractRedisInstance(port) {
  var executable: File = RedisExecProvider.defaultProvider.get
  this.args = util.Arrays.asList(executable.getAbsolutePath,
    "--port", Integer.toString(port.intValue))

  def this(executable: File, port: Integer) {
    this(port)
    this.executable = executable
    this.args = util.Arrays.asList(executable.getAbsolutePath,
      "--port", Integer.toString(port.intValue))
  }

  def this(redisExecProvider: RedisExecProvider, port: Integer) {
    this(port)
    this.executable = redisExecProvider.get.getAbsoluteFile
    this.args = util.Arrays.asList(redisExecProvider.get.getAbsolutePath,
      "--port", Integer.toString(port.intValue))
  }

  override protected def redisReadyPattern: String = ".*Ready to accept connections.*"
}
