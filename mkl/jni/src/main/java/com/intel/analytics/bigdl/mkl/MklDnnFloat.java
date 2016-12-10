package com.intel.analytics.bigdl.mkl;

/**
 * MKL DNN Library Wrapper for JVM
 */
public class MklDnnFloat {
    static {
        MKL.isMKLLoaded();
    }
    public native static long layoutCreate(int dimension,
                                           long[] size,
                                           long[] strides);
    public native static long layoutCreateFromPrimitive(long primitive,
                                                        int type);

    public native static long layoutGetMemorySize(long layout);

    /**
     * compare two layouts and return the result.
     * @param layout1
     * @param layout2
     * @return If layouts are the same, return 1.
     */
    public native static int layoutCompare(long layout1,
                                           long layout2);

    public native static void deletePrimitive(long primitive);

    public native static long allocateBuffer();
    public native static void releaseBuffer();

    /**
     * create conversion primitive from layout1 to layout2
     * @param layout1 src layout
     * @param layout2 dst layout
     * @return
     */
    public native static long conversionCreate(long layout1,
                                               long layout2);
    public native static void conversionExecuteToUsr(float[] usr,
                                                     int usrOffset,
                                                     float[] mkl,
                                                     long primitive);
    public native static void conversionExecuteToMkl(float[] usr,
                                                     int usrOffset,
                                                     float[] mkl,
                                                     long primitive);

    // convolution wrapper
    public native static long convolutionCreateForward(int algorithm,
                                                       long groups,
                                                       long dimension,
                                                       long[] inputSize,
                                                       long[] outputSize,
                                                       long[] weightSize,
                                                       long[] strides,
                                                       int [] pad,
                                                       int boderType);
    public native static long convolutionCreateBackwardData(int algorithm,
                                                            long groups,
                                                            long dimension,
                                                            long[] inputSize,
                                                            long[] outputSize,
                                                            long[] weightSize,
                                                            long[] strides,
                                                            int[] pad,
                                                            int borderType);
    public native static long convolutionCreateBackwardKernel(int algorithm,
                                                              long groups,
                                                              long dimension,
                                                              long[] inputSize,
                                                              long[] outputSize,
                                                              long[] weightSize,
                                                              long[] strides,
                                                              int[] pad,
                                                              int borderType);
    public native static long convolutionCreateBackwardBias(int algorithm,
                                                            long groups,
                                                            long dimension,
                                                            long[] outputSize);
    public native static void convolutionForwardExecute(float[] input,
                                                        float[] weight,
                                                        float[] bias,
                                                        float[] output,
                                                        long primitive);
    public native static void convolutionBackwardDataExecute(float[] gradInput,
                                                             float[] gradOutput,
                                                             float[] backWeight,
                                                             long primitive);
    public native static void convolutionBackwardKernelExecute(float[] input,
                                                               float[] gradOutput,
                                                               float[] gradWeight,
                                                               long primitive);
    public native static void convolutionBackwardBiasExecute(float[] gradOutput,
                                                             float[] gradBias,
                                                             long primitive);

    // ReLU
    public native static long reluCreateForward(long layout, float nagtiveSlope);
    public native static long reluCreateBackward(long layout1, long layout2, float nagtiveSlope);
    public native static void reluForwardExecute(float[] input, float[] output, long primitive);
    public native static void reluBackwardExecute(float[] input,
                                                  float[] gradInput,
                                                  float[] gradOutput,
                                                  long primitive);

    // Pooling
    public native static long poolCreateForward(int algorithm,
                                                long layout,
                                                long[] kernelSize,
                                                long[] stride,
                                                int[] pad,
                                                int borderType);
    public native static long poolCreateBackward(int algorithm,
                                                 long layout,
                                                 long[] kernelSize,
                                                 long[] stride,
                                                 int[] pad,
                                                 int borderType);
    public native static void poolForwardExecute(float[] input,
                                                 float[] output,
                                                 float[] workspace,
                                                 long primitive);
    public native static void poolBackwardExecute(float[] gradInput,
                                                  float[] gradOutput,
                                                  float[] workspace,
                                                  long primitive);

    // Linear
    public native static long linearCreateForwardBias(long dimension,
                                                      long[] inputSize,
                                                      long outputChannel);
    public native static long linearCreateBackData(long dimension,
                                                   long[] inputSize,
                                                   long outputChannel);
    public native static long linearCreateBackWeight(long dimension,
                                                     long[] inputSize,
                                                     long outputChannel);
    public native static long linearCreateBackBias(long dimension,
                                                   long[] outputSize);

    public native static void linearForwardExecute(float[] input,
                                                   float[] weight,
                                                   float[] bias,
                                                   float[] output,
                                                   long primitive);
    public native static void linearBackDataExecute(float[] gradInput,
                                                    float[] gradOutput,
                                                    float[] weight,
                                                    long primitive);
    public native static void linearBackWeightExecute(float[] input,
                                                      float[] gradOutput,
                                                      float[] gradWeight,
                                                      long primitive);
    public native static void linearBackBiasExecute(float[] gradOutput,
                                                    float[] gradBias,
                                                    long primitive);

    // LRN
    public native static long lrnCreateForward(long layout,
                                               long size,
                                               float alpha,
                                               float beta,
                                               float k);
    public native static long lrnCreateBackward(long layout1,
                                                long layout2,
                                                long size,
                                                float alpha,
                                                float beta,
                                                float k);
    public native static void lrnForwardExecute(float[] input,
                                                float[] output,
                                                float[] workspace,
                                                long primitive);
    public native static void lrnBackwardExecute(float[] input,
                                                 float[] gradInput,
                                                 float[] gradOutput,
                                                 float[] workspace,
                                                 long primitive);
}
