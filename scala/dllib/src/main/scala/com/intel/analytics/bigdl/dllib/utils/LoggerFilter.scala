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

package com.intel.analytics.bigdl.utils

import org.apache.log4j._
import java.nio.file.{Paths, Files}
import scala.collection.JavaConverters._

/**
 * logger filter, which will filter the log of Spark(org, breeze, akka) to file.
 * it could be set by user through `-Dbigdl.utils.LoggerFilter.logFile`
 */
object LoggerFilter {

  private val pattern = "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n"

  /**
   * redirect log to `filePath`
   *
   * @param filePath log file path
   * @param level logger level, the default is Level.INFO
   * @return a new file appender
   */
  private def fileAppender(filePath: String, level: Level = Level.INFO): FileAppender = {
    val fileAppender = new FileAppender
    fileAppender.setName("FileLogger")
    fileAppender.setFile(filePath)
    fileAppender.setLayout(new PatternLayout(pattern))
    fileAppender.setThreshold(level)
    fileAppender.setAppend(true)
    fileAppender.activateOptions()

    fileAppender
  }

  /**
   * redirect log to console or stdout
   *
   * @param level logger level, the default is Level.INFO
   * @return a new console appender
   */
  private def consoleAppender(level: Level = Level.INFO): ConsoleAppender = {
    val console = new ConsoleAppender
    console.setLayout(new PatternLayout(pattern))
    console.setThreshold(level)
    console.activateOptions()
    console.setTarget("System.out")

    console
  }

  /**
   * find the logger of `className` and add a new appender to it.
   *
   * @param className class which user defined
   * @param appender appender, eg. return of `fileAppender` or `consoleAppender`
   */
  private def classLogToAppender(className: String, appender: Appender): Unit = {
    Logger.getLogger(className).addAppender(appender)
  }

  /**
   * 1. redirect all logs of Spark to file, which can be set by `-Dbigdl.utils.LoggerFilter.logFile`
   * the default file is under current workspace named `bigdl.log`.
   * 2. `-Dbigdl.utils.LoggerFilter.disable=true` will disable redirection.
   * and add an console appender for `com.intel.analytics.bigdl.optim`, because we set the threshold
   * to ERROR first.
   */
  def redirectSparkInfoLogs(): Unit = {
    val disable = System.getProperty("bigdl.utils.LoggerFilter.disable", "false")
    if (disable.equalsIgnoreCase("false")) {
      val optimClass = "com.intel.analytics.bigdl.optim"
      val default = Paths.get(System.getProperty("user.dir"), "bigdl.log").toString
      val logFile = System.getProperty("bigdl.utils.LoggerFilter.logFile", default)

      // If file doesn't exist, create a new one. If it's a directory, throw an error.
      println(logFile)
      val logFilePath = Paths.get(logFile)
      if (!Files.exists(logFilePath)) {
        Files.createFile(logFilePath)
      } else if (Files.isDirectory(logFilePath)) {
        Logger.getLogger(getClass)
          .error(s"$logFile exists and is an directory. Can't redirect to it.")
      }

      val defaultClasses = List("org", "akka", "breeze")

      for (clz <- defaultClasses) {
        classLogToAppender(clz, consoleAppender(Level.ERROR))
        Logger.getLogger(clz).setAdditivity(false)
      }

      for (clz <- optimClass :: defaultClasses) {
        classLogToAppender(clz, fileAppender(logFile, Level.INFO))
      }

      classLogToAppender(optimClass, consoleAppender(Level.INFO))
    }
  }
}
