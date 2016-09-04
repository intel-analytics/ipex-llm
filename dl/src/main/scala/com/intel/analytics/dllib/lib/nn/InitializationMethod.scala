package com.intel.analytics.dllib.lib.nn

sealed trait InitializationMethod

case object Default extends InitializationMethod
case object Xavier extends InitializationMethod
