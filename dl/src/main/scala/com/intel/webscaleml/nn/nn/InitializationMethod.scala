package com.intel.webscaleml.nn.nn

sealed trait InitializationMethod

case object Default extends InitializationMethod
case object Xavier extends InitializationMethod
