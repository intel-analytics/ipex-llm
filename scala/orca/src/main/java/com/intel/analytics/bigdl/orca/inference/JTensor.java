package com.intel.analytics.zoo.pipeline.inference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class JTensor {
  private List<Float> data;
  private List<Integer> shape;

  public JTensor() {
  }

  public JTensor(List<Float> data, List<Integer> shape) {
    this.data = data;
    this.shape = shape;
  }

  public JTensor(List<Float> data, Integer[] shape) {
    this.data = data;
    this.shape = Arrays.asList(shape);
  }

  public JTensor(List<Float> data, int[] shape) {
    this.data = data;
    this.shape = new ArrayList<Integer>();
    for(int s : shape) {
      this.shape.add(s);
    }
  }

  public List<Float> getData() {
    return data;
  }

  public void setData(List<Float> data) {
    this.data = data;
  }

  public List<Integer> getShape() {
    return shape;
  }

  public void setShape(List<Integer> shape) {
    this.shape = shape;
  }

  @Override
  public String toString() {
    return "JTensor{" +
            "data=" + data +
            ", shape=" + shape +
            '}';
  }
}
