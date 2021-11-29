/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib;

public class ClusteringIterationStatsVector {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected ClusteringIterationStatsVector(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(ClusteringIterationStatsVector obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_ClusteringIterationStatsVector(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public ClusteringIterationStatsVector() {
    this(swigfaissJNI.new_ClusteringIterationStatsVector(), true);
  }

  public void push_back(ClusteringIterationStats arg0) {
    swigfaissJNI.ClusteringIterationStatsVector_push_back(swigCPtr, this, ClusteringIterationStats.getCPtr(arg0), arg0);
  }

  public void clear() {
    swigfaissJNI.ClusteringIterationStatsVector_clear(swigCPtr, this);
  }

  public ClusteringIterationStats data() {
    long cPtr = swigfaissJNI.ClusteringIterationStatsVector_data(swigCPtr, this);
    return (cPtr == 0) ? null : new ClusteringIterationStats(cPtr, false);
  }

  public long size() {
    return swigfaissJNI.ClusteringIterationStatsVector_size(swigCPtr, this);
  }

  public ClusteringIterationStats at(long n) {
    return new ClusteringIterationStats(swigfaissJNI.ClusteringIterationStatsVector_at(swigCPtr, this, n), true);
  }

  public void resize(long n) {
    swigfaissJNI.ClusteringIterationStatsVector_resize(swigCPtr, this, n);
  }

  public void swap(ClusteringIterationStatsVector other) {
    swigfaissJNI.ClusteringIterationStatsVector_swap(swigCPtr, this, ClusteringIterationStatsVector.getCPtr(other), other);
  }

}
