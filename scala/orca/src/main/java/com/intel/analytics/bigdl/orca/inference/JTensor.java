package com.intel.analytics.bigdl.orca.inference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class JTensor {
  private float[] data;
  private int[] shape;

  public JTensor() {
  }

  public JTensor(List<Float> data, List<Integer> shape) {
    this.data = new float[data.size()];
    for (int i = 0; i < data.size(); i++){
      this.data[i] = data.get(i);
    }
    this.shape = new int[shape.size()];
    for (int i = 0; i < shape.size(); i++){
      this.shape[i] = shape.get(i);
    }
  }

  public JTensor(List<Float> data, Integer[] shape) {
    this.data = new float[data.size()];
    for (int i = 0; i < data.size(); i++){
      this.data[i] = data.get(i);
    }
    this.shape = new int[shape.length];
    for (int i = 0; i < shape.length; i++){
      this.shape[i] = shape[i];
    }
  }

  public JTensor(List<Float> data, int[] shape) {
    this.data = new float[data.size()];
    for (int i = 0; i < data.size(); i++){
      this.data[i] = data.get(i);
    }
    this.shape = new int[shape.length];
    for (int i = 0; i < shape.length; i++){
      this.shape[i] = shape[i];
    }
  }

  public JTensor(float[] data, List<Integer> shape) {
    this.data = new float[data.length];
    for (int i = 0; i < data.length; i++){
      this.data[i] = data[i];
    }
    this.shape = new int[shape.size()];
    for (int i = 0; i < shape.size(); i++){
      this.shape[i] = shape.get(i);
    }
  }

  public JTensor(float[] data, Integer[] shape) {
    this.data = new float[data.length];
    for (int i = 0; i < data.length; i++){
      this.data[i] = data[i];
    }
    this.shape = new int[shape.length];
    for (int i = 0; i < shape.length; i++){
      this.shape[i] = shape[i];
    }
  }

  public JTensor(float[] data, int[] shape) {
    this.data = new float[data.length];
    for (int i = 0; i < data.length; i++){
      this.data[i] = data[i];
    }
    this.shape = new int[shape.length];
    for (int i = 0; i < shape.length; i++){
      this.shape[i] = shape[i];
    }
  }

  public JTensor(float[] data, int[] shape, boolean copy){
    if (copy) {
      this.data = new float[data.length];
      for (int i = 0; i < data.length; i++){
        this.data[i] = data[i];
      }
      this.shape = new int[shape.length];
      for (int i = 0; i < shape.length; i++){
        this.shape[i] = shape[i];
      }
    }
    else {
      this.data = data;
      this.shape = shape;
    }
  }

  public float[] getData() {
    return data;
  }

  public void setData(float[] data) {
    this.data = data;
  }

  public int[] getShape() {
    return shape;
  }

  public void setShape(int[] shape) {
    this.shape = shape;
  }

  public static String toString(int[] a) {
    if (a == null)
      return "null";
    int iMax = a.length - 1;
    if (iMax == -1)
      return "[]";

    int max = Math.min(500, iMax);

    StringBuilder b = new StringBuilder();
    b.append('[');
    for (int i = 0; ; i++) {
      b.append(a[i]);
      if (i == max && i < iMax)
        return b.append(", ... ]").toString();
      if (i == iMax)
        return b.append(']').toString();
      b.append(", ");
    }
  }

  public static String toString(float[] a) {
    if (a == null)
      return "null";
    int iMax = a.length - 1;
    if (iMax == -1)
      return "[]";

    int max = Math.min(500, iMax);

    StringBuilder b = new StringBuilder();
    b.append('[');
    for (int i = 0; ; i++) {
      b.append(a[i]);
      if (i == max && i < iMax)
        return b.append(", ... ]").toString();
      if (i == iMax)
        return b.append(']').toString();
      b.append(", ");
    }
  }

  @Override
  public String toString() {
    return "JTensor{" +
            "data=" + toString(data) +
            ", shape=" + toString(shape) +
            '}';
  }
}