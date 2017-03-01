// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorboard/src/summary.proto

package org.tensorflow.framework;

public interface SummaryDescriptionOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorboard.SummaryDescription)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Hint on how plugins should process the data in this series.
   * Supported values include "scalar", "histogram", "image", "audio"
   * </pre>
   *
   * <code>optional string type_hint = 1;</code>
   */
  java.lang.String getTypeHint();
  /**
   * <pre>
   * Hint on how plugins should process the data in this series.
   * Supported values include "scalar", "histogram", "image", "audio"
   * </pre>
   *
   * <code>optional string type_hint = 1;</code>
   */
  com.google.protobuf.ByteString
      getTypeHintBytes();
}
