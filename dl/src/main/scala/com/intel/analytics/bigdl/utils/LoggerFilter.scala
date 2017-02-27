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

package com.intel.analytics.bigdl.utils

import org.apache.log4j._

import scala.collection.JavaConverters._

/**
 * logger appender, which will output the log to files
 */
object LoggerFilter {

  private val pattern = "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n"

  /**
   * redirect log to `filePath`, whose default value is /tmp/bigdl.log
   *
   * @param filePath log file path, default value is /tmp/bigdl.log
   * @return a new file appender
   */
  private def fileAppender(filePath: String = "/tmp/bigdl.log",
    level: Level = Level.INFO): FileAppender = {
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
   * find the logger of `clz` and add a new appender to it.
   *
   * @param clz class which user defined
   * @param appender appender, eg. return of `fileAppender` or `consoleAppender`
   */
  private def classLogToAppender(clz: String, appender: Appender): Unit = {
    Logger.getLogger(clz).addAppender(appender)
  }

  /**
   * make the level of console appender in root logger to ERROR,
   * which will interfere with all other loggers.
   */
  private def setConsoleLevelToError(): Unit = {
    val appenders = Logger.getRootLogger.getAllAppenders.asScala

    for (appender <- appenders) {
      appender match {
        case a: ConsoleAppender => a.setThreshold(Level.ERROR)
        case _ => // ignore other appender except ConsoleAppender
      }
    }
  }

  /**
   * redirect all logs of Spark to fileAppender (/tmp/bigdl.log).
   * and add an console appender for `com.intel.analytics.bigdl.optim`
   */
  def redirectSparkInfoLogs(): Unit = {
    setConsoleLevelToError()

    val optimClz = "com.intel.analytics.bigdl.optim"

    for (clz <- List("org", "akka", "breeze", optimClz)) {
      LoggerFilter.classLogToAppender(clz, LoggerFilter.fileAppender())
    }

    LoggerFilter.classLogToAppender(optimClz, LoggerFilter.consoleAppender())
  }
}
