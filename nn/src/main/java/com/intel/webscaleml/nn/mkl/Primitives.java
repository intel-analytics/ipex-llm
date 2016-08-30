package com.intel.webscaleml.nn.mkl;


public class Primitives {
    static {
        System.loadLibrary("jmkl_dnn");
    }

    public native static void relu_forward(float[] input, float[] output, int y, int x, int f, int b);
    public native static void relu_backward(float[] input, float[] grad_output, float[] grad_input, int y, int x, int f, int b);

    public native static void convolution_forward(
    float[] input, int input_offset, int input_b, int input_z, int input_y, int input_x,
    float[] output, int output_offset, int output_b, int output_z, int output_y, int output_x,
    float[] weight, int weight_offset,
    float[] bias, int bias_offset,
    int stride_b, int stride_f, int stride_y, int stride_x,
    int conv_size_y, int conv_size_x, int conv_size_ofm, int conv_size_ifm);


    public native static void convolution_backward(
    float[] bw_output, int bw_output_offset, int output_b, int output_z, int output_y, int output_x,
    float[] weights_diff, int weights_diff_offset, float[] biases_diff, int biases_diff_offset,
    float[] bw_input, int bw_input_offset, int input_b, int input_z, int input_y, int input_x,
    float[] fw_input, int fw_input_offset, float[] weight, int weight_offset,
    float[] bias, int bias_offset, int out_siz_z,
    int stride_b, int stride_z, int stride_y, int stride_x,
    int conv_size_b, int conv_size_z, int conv_size_y, int conv_size_x);

    public native static void setNumThreads(int numThreads);
}
