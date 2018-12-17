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

 // scalastyle:off
 // | Property Name                           | Default            | Meaning                                      |
 // |-----------------------------------------+--------------------+----------------------------------------------|
 // | bigdl.utils.LoggerFilter.disable        | false              | Disable redirecting logs of Spark and BigDL. |
 // |                                         |                    | Output location depends on log4j.properties  |
 // | bigdl.utils.LoggerFilter.logFile        | user.dir/bigdl.log | The log file user defined.                   |
 // | bigdl.utils.LoggerFilter.enableSparkLog | true               | Enable redirecting logs of Spark to logFile  |
 // scalastyle:on

/**
 * logger filter, which will filter the log of Spark(org, breeze, akka) to file.
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

  private val defaultPath = Paths.get(System.getProperty("user.dir"), "bigdl.log").toString

  /**
   * 1. redirect all spark log to file, which can be set by `-Dbigdl.utils.LoggerFilter.logFile`
   *    the default file is under current workspace named `bigdl.log`.
   * 2. `-Dbigdl.utils.LoggerFilter.disable=true` will disable redirection.
   * 3. `-Dbigdl.utils.LoggerFilter.enableSparkLog=false` will not output spark log to file
   */
  def redirectSparkInfoLogs(logPath: String = defaultPath): Unit = {
    val disable = System.getProperty("bigdl.utils.LoggerFilter.disable", "false")
    val enableSparkLog = System.getProperty("bigdl.utils.LoggerFilter.enableSparkLog", "true")

    def getLogFile: String = {
      val logFile = System.getProperty("bigdl.utils.LoggerFilter.logFile", logPath)

      // If the file doesn't exist, create a new one. If it's a directory, throw an error.
      val logFilePath = Paths.get(logFile)
      if (!Files.exists(logFilePath)) {
        Files.createFile(logFilePath)
      } else if (Files.isDirectory(logFilePath)) {
        Logger.getLogger(getClass)
          .error(s"$logFile exists and is an directory. Can't redirect to it.")
      }

      logFile
    }

    if (disable.equalsIgnoreCase("false")) {
      val logFile = getLogFile

      val defaultClasses = List("org", "akka", "breeze")

      for (clz <- defaultClasses) {
        classLogToAppender(clz, consoleAppender(Level.ERROR))
        Logger.getLogger(clz).setAdditivity(false)
      }
      // it should be set to WARN for the progress bar
      Logger.getLogger("org.apache.spark.SparkContext").setLevel(Level.WARN)

      // set all logs to file
      Logger.getRootLogger.addAppender(fileAppender(logFile, Level.INFO))

      // because we have set all defaultClasses loggers additivity to false
      // so we should reconfigure them.
      if (enableSparkLog.equalsIgnoreCase("true")) {
        for (clz <- defaultClasses) {
          classLogToAppender(clz, fileAppender(logFile, Level.INFO))
        }
      }
    }
  }
}
