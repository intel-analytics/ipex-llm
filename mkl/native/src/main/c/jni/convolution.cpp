#include <jni.h>

#include "debug.h"
#include "layer.h"
#include "memory.h"
#include "utils.h"

#include <iostream>
#include <fstream>

static int getMKLBuildDate()
{
  static int build = 0;
  if (build == 0) {
    MKLVersion v;
    mkl_get_version(&v);
    build = atoi(v.Build);
  }
  return build;
}

template <typename DType>
class MKLConvolution : public MKLLayer<DType>
{
 public:
  MKLConvolution();
  ~MKLConvolution();

  void init(size_t inputNumber, size_t inputChannel, size_t inputHeight,
            size_t inputWidth, size_t kernelNumber, size_t kernelChannel,
            size_t kernelHeight, size_t kernelWidth, size_t strideHeight,
            size_t strideWidth, int padHeight, int padWidth, int dimension,
            int groups, const char *name);

  void updateOutput(DType *input, DType *output);
  void updateGradInput(DType *input, DType *gradOutput, DType *gradInput);
  void updateGradKernel(DType *input, DType *gradOutput, DType *gradKernel);
  void updateGradBias(DType *input, DType *gradOutput, DType *gradBias);

  std::shared_ptr<MKLData<DType>> kernel;
  /*
   * Attention 2016-10-10
   *
   * I don't know why should we must set different kernel parameters
   * for forward and backward (updateOutput and updateGradInput).
   * Otherwise, the result of gradient input is not correct.
   */
  std::shared_ptr<MKLData<DType>> backKernel;
  std::shared_ptr<MKLData<DType>> bias;

  std::shared_ptr<MKLData<DType>> gradKernel;
  std::shared_ptr<MKLData<DType>> gradBias;

  std::shared_ptr<MKLData<DType>> gradOutputK;
  std::shared_ptr<MKLData<DType>> gradOutputB;

 private:
  // this method is not the same as createMklLayout in MKLMemory
  void firstPass();
  void preExecute(DType *input);

  DType *kernelAdr;
  DType *biasAdr;

  dnnPrimitive_t kernelPrim, biasPrim;

  size_t groups;

  size_t inputSize[4];
  size_t inputStrides[4];

  size_t outputSize[4];
  size_t outputStrides[4];

  size_t kernelDimension;
  size_t kernelSize[5];
  size_t kernelStrides[5];

  size_t biasSize[1];
  size_t biasStrides[1];

  size_t stride[2];
  int pad[2];
};

template <typename DType>
MKLConvolution<DType>::MKLConvolution()
    : kernel(new MKLData<DType>),
      backKernel(new MKLData<DType>),
      bias(new MKLData<DType>),
      gradKernel(new MKLData<DType>),
      gradBias(new MKLData<DType>),
      kernelAdr(NULL),
      biasAdr(NULL),
      kernelPrim(NULL),
      biasPrim(NULL),
      gradOutputK(new MKLData<DType>),
      gradOutputB(new MKLData<DType>)
{
}

template <typename DType>
MKLConvolution<DType>::~MKLConvolution()
{
  dnnDelete<DType>(kernelPrim);
  dnnDelete<DType>(biasPrim);
}

template <typename DType>
void MKLConvolution<DType>::init(size_t inputNumber, size_t inputChannel,
                                 size_t inputHeight, size_t inputWidth,
                                 size_t kernelNumber, size_t kernelChannel,
                                 size_t kernelHeight, size_t kernelWidth,
                                 size_t strideHeight, size_t strideWidth,
                                 int padHeight, int padWidth, int dimension,
                                 int groups, const char *name)
{
  this->dimension = dimension;
  this->groups    = groups;
  this->name.assign(name);

  inputSize[0] = inputWidth;
  inputSize[1] = inputHeight;
  inputSize[2] = inputChannel;
  inputSize[3] = inputNumber;

  inputStrides[0] = 1;
  for (int i        = 1; i < 4; i++)
    inputStrides[i] = inputStrides[i - 1] * inputSize[i - 1];

  size_t outputWidth =
      computeOut(inputWidth, padWidth, kernelWidth, strideWidth, false);
  size_t outputHeight =
      computeOut(inputHeight, padHeight, kernelHeight, strideHeight, false);

  // the output channel is as same as the number of kernel.
  // and the output number must be as same as the number of input too.
  outputSize[0] = outputWidth;
  outputSize[1] = outputHeight;
  outputSize[2] = kernelNumber;
  outputSize[3] = inputNumber;

  outputStrides[0] = 1;
  for (int i         = 1; i < 4; i++)
    outputStrides[i] = outputStrides[i - 1] * outputSize[i - 1];

  // comes from IntelCaffe.
  size_t groupsMKL = groups;
  kernelDimension  = this->dimension + (groups != 1);
  if (getMKLBuildDate() < 20160701) {
    kernelDimension = this->dimension;
    groupsMKL       = 1;
  }

  kernelSize[0] = kernelWidth;
  kernelSize[1] = kernelHeight;
  kernelSize[2] = kernelChannel / groups;
  kernelSize[3] = kernelNumber / groupsMKL;
  kernelSize[4] = groupsMKL;

  for (int i = 0; i < 5; i++) {
    LOG(INFO) << "kernelSize[" << i << "] = " << kernelSize[i];
  }

  kernelStrides[0] = 1;
  for (int i         = 1; i < 5; i++)
    kernelStrides[i] = kernelStrides[i - 1] * kernelSize[i - 1];

  biasSize[0]    = kernelNumber;
  biasStrides[0] = 1;

  stride[0] = strideWidth;
  stride[1] = strideHeight;

  pad[0] = -padWidth;
  pad[1] = -padHeight;

  // create usr layout
  this->input->createUsrLayout(dimension, inputSize, inputStrides);
  this->output->createUsrLayout(dimension, outputSize, outputStrides);
  this->kernel->createUsrLayout(kernelDimension, kernelSize, kernelStrides);
  this->backKernel->createUsrLayout(kernelDimension, kernelSize, kernelStrides);
  this->bias->createUsrLayout(1, biasSize, biasStrides);

  this->gradInput->createUsrLayout(dimension, inputSize, inputStrides);
  this->gradOutput->createUsrLayout(dimension, outputSize, outputStrides);
  this->gradKernel->createUsrLayout(kernelDimension, kernelSize, kernelStrides);
  // bias dimension is 1
  this->gradBias->createUsrLayout(1, biasSize, biasStrides);

  this->gradOutputK->createUsrLayout(dimension, outputSize, outputStrides);
  this->gradOutputB->createUsrLayout(dimension, outputSize, outputStrides);
}

template <typename DType>
void MKLConvolution<DType>::firstPass()
{
  dnnError_t status = E_UNIMPLEMENTED;
  // forward
  status = dnnGroupsConvolutionCreateForwardBias<DType>(
      &(this->forwardPrim), NULL, dnnAlgorithmConvolutionDirect, groups,
      this->dimension, inputSize, outputSize, kernelSize, stride, pad,
      dnnBorderZeros);
  CHECK_EQ(status, E_SUCCESS);

  this->input->createMklLayout(this->forwardPrim, dnnResourceSrc);
  this->output->createMklLayout(this->forwardPrim, dnnResourceDst);
  this->kernel->createMklLayout(this->forwardPrim, dnnResourceFilter);
  this->bias->createMklLayout(this->forwardPrim, dnnResourceBias);

  // backward data
  status = dnnGroupsConvolutionCreateBackwardData<DType>(
      &(this->backwardPrim), NULL, dnnAlgorithmConvolutionDirect, groups,
      this->dimension, inputSize, outputSize, kernelSize, stride, pad,
      dnnBorderZeros);
  CHECK_EQ(status, E_SUCCESS);

  this->gradOutput->createMklLayout(this->backwardPrim, dnnResourceDiffDst);
  this->gradInput->createMklLayout(this->backwardPrim, dnnResourceDiffSrc);
  this->backKernel->createMklLayout(this->backwardPrim, dnnResourceFilter);

  // backward kernel
  status = dnnGroupsConvolutionCreateBackwardFilter<DType>(
      &kernelPrim, NULL, dnnAlgorithmConvolutionDirect, groups, this->dimension,
      inputSize, outputSize, kernelSize, stride, pad, dnnBorderZeros);
  CHECK_EQ(status, E_SUCCESS);

  this->gradKernel->createMklLayout(this->kernelPrim, dnnResourceDiffFilter);
  this->gradOutputK->createMklLayout(this->kernelPrim, dnnResourceDiffDst);

  // backward bias
  status = dnnGroupsConvolutionCreateBackwardBias<DType>(
      &biasPrim, NULL, dnnAlgorithmConvolutionDirect, groups, this->dimension,
      outputSize);
  CHECK_EQ(status, E_SUCCESS);

  this->gradBias->createMklLayout(this->biasPrim, dnnResourceDiffBias);
  this->gradOutputB->createMklLayout(this->biasPrim, dnnResourceDiffDst);

  // we create the layout only at the first time
  this->isFirstPass = false;
}

template <typename DType>
void MKLConvolution<DType>::preExecute(DType *input)
{
  if (this->getIsUseOpenMp()) {
    caffe::cpu::OpenMpManager::setGpuDisabled();
    caffe::cpu::OpenMpManager::bindOpenMpThreads();
  }

  this->input->createConversion();
  //LOG(DBG) << "DOES INPUT CREATE NEW MEM?";
  this->kernel->createConversion();
  //LOG(DBG) << "AFTER KERNEL";
  this->bias->createConversion();
  //LOG(DBG) << "AFTER BIAS";
}

template <typename DType>
void MKLConvolution<DType>::updateOutput(DType *input, DType *output)
{
  if (this->isFirstPass) firstPass();

  // Because the address will change every time, so we need create conversion
  // every forward/backward.
  // TODO Should we set the kernel and bias address every time?
  preExecute(input);
  this->output->createConversion();
  // this->output->setZero();
  //LOG(DBG) << "AFTER OUTPUT";

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->input->getUsrData()),
                   this->inputSize[3], this->inputSize[2], this->inputSize[1],
                   this->inputSize[0], "Forward input");
#endif

  dnnError_t status;
  void *resources[dnnResourceNumber];

  resources[dnnResourceFilter] = this->kernel->getConvertedData();
  resources[dnnResourceBias]   = this->bias->getConvertedData();
  resources[dnnResourceSrc]    = this->input->getConvertedData();
  resources[dnnResourceDst]    = this->output->getData();

  PERFSTART();
  status = dnnExecute<DType>(this->forwardPrim, resources);
  PERFEND("main computing");
  CHECK_EQ(status, E_SUCCESS);

  this->input->setIsConverted(true);

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->output->getData()),
                   outputSize[3], outputSize[2], outputSize[1], outputSize[0],
                   "Forward output");
#endif

  if (!this->output->isUseNext()) {
    this->output->backToUsr();
  }
}

template <typename DType>
void MKLConvolution<DType>::updateGradInput(DType *input, DType *gradOutput,
                                            DType *gradInput)
{
  dnnError_t status;
  void *resources[dnnResourceNumber];

  preExecute(input);

  this->gradOutput->createConversion();
  this->gradInput->createConversion();
  this->backKernel->createConversion();

  resources[dnnResourceDiffDst] = this->gradOutput->getConvertedData();
  resources[dnnResourceFilter]  = this->backKernel->getConvertedData();
  resources[dnnResourceDiffSrc] = this->gradInput->getData();

  //LOG(DBG) << "resources[dnnResourceDiffDst] " << resources[dnnResourceDiffDst];

  // 4. main computing parts.
  PERFSTART();
  status = dnnExecute<DType>(this->backwardPrim, resources);
  CHECK_EQ(status, E_SUCCESS);
  PERFEND("main computing");

  this->gradOutput->setIsConverted(true);

  if (!this->gradInput->isUsePrev()) {
    this->gradInput->backToUsr();
  }

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->gradInput->getUsrData()),
                   inputSize[3], inputSize[2], inputSize[1], inputSize[0],
                   "backward gradient input");
#endif
}

template <typename DType>
void MKLConvolution<DType>::updateGradKernel(DType *input, DType *gradOutput,
                                             DType *gradKernel)
{
  dnnError_t status;
  void *resources[dnnResourceNumber];

  preExecute(input);

  this->gradOutputK->layoutNext = this->gradOutput->layoutNext;
  this->gradOutputK->dataNext = this->gradOutput->dataNext;
  if (this->gradOutput->isUseNext()) {
    this->gradOutputK->setUseNext(true);
  }

  this->gradOutputK->createConversion();
  this->gradKernel->createConversion();

  resources[dnnResourceDiffDst]    = this->gradOutputK->getConvertedData();
  resources[dnnResourceSrc]        = this->input->getConvertedData();
  resources[dnnResourceDiffFilter] = this->gradKernel->getData();

  // 4. main computing parts.
  PERFSTART();
  status = dnnExecute<DType>(this->kernelPrim, resources);
  CHECK_EQ(status, E_SUCCESS);
  PERFEND("main computing");

  this->input->setIsConverted(false);
  // because we may not do upgradInput at the first layer of network,
  // so the kernel converted attribute should be set to false here.
  // and gradOutput converted attributes should be set to true here,
  // which MUST be set to false back at updateGradBias.
  this->gradOutput->setIsConverted(true);

  // we don't need kernel at all here, we use backKernel!
  // this->kernel->setIsConverted(false);

  // the kernel need not re-use for previous layer
  this->gradKernel->backToUsr();
}

template <typename DType>
void MKLConvolution<DType>::updateGradBias(DType *input, DType *gradOutput,
                                           DType *gradBias)
{
  dnnError_t status;
  void *resources[dnnResourceNumber];

  preExecute(input);

  if (this->gradOutput->isUseNext()) {
    this->gradOutputB->layoutNext = this->gradOutput->layoutNext;
    this->gradOutputB->dataNext = this->gradOutput->dataNext;
    this->gradOutputB->setUseNext(true);
  }

  this->gradOutputB->createConversion();
  this->gradBias->createConversion();

  resources[dnnResourceDiffDst]  = this->gradOutputB->getConvertedData();
  resources[dnnResourceDiffBias] = this->gradBias->getData();

  // 4. main computing parts.
  PERFSTART();
  status = dnnExecute<DType>(this->biasPrim, resources);
  CHECK_EQ(status, E_SUCCESS);
  PERFEND("main computing");

  this->gradOutput->setIsConverted(false);

  this->gradBias->backToUsr();
}

template <typename ArrayType, typename DType>
jlong JNIConvolutionInit(JNIEnv *env, jclass thisClass, jint inputNumber,
                         jint inputChannel, jint inputHeight, jint inputWidth,
                         jint kernelNumber, jint kernelChannel,
                         jint kernelHeight, jint kernelWidth, jint strideHeight,
                         jint strideWidth, jint padHeight, jint padWidth,
                         jint dimension, jint groups, const jstring name)
{
  const char *jName = env->GetStringUTFChars(name, NULL);
  MKLConvolution<DType> *conv = new MKLConvolution<DType>();
  conv->init(inputNumber, inputChannel, inputHeight, inputWidth, kernelNumber,
             kernelChannel, kernelHeight, kernelWidth, strideHeight,
             strideWidth, padHeight, padWidth, dimension, groups, jName);

  return reinterpret_cast<long>(conv);
}

template <typename ArrayType, typename DType>
void JNIConvolutionUpdateOutput(JNIEnv *env, jclass thisClass, ArrayType input,
                                jint inputOffset, ArrayType output,
                                jint outputOffset, ArrayType kernel,
                                jint kernelOffset, ArrayType bias,
                                jint biasOffset, long classPtr)
{
  MKLConvolution<DType> *ptr =
      reinterpret_cast<MKLConvolution<DType> *>(classPtr);

  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutput(
      new ZipArray<ArrayType, DType>(env, output, outputOffset, ptr->output));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernel(
      new ZipArray<ArrayType, DType>(env, kernel, kernelOffset, ptr->kernel));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBias(
      new ZipArray<ArrayType, DType>(env, bias, biasOffset, ptr->bias));

  ptr->updateOutput(jInput->getPtr(), jOutput->getPtr());
}

template <typename ArrayType, typename DType>
void JNIConvolutionUpdateGradInput(JNIEnv *env, jclass thisClass,
                                   ArrayType input, jint inputOffset,
                                   ArrayType outputDiff, jint outputDiffOffset,
                                   ArrayType inputDiff, jint inputDiffOffset,
                                   ArrayType kernel, jint kernelOffset,
                                   ArrayType bias, jint biasOffset,
                                   long classPtr)
{
  MKLConvolution<DType> *ptr =
      reinterpret_cast<MKLConvolution<DType> *>(classPtr);
  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutputDiff(
      new ZipArray<ArrayType, DType>(env, outputDiff, outputDiffOffset,
                                     ptr->gradOutput));

  std::shared_ptr<ZipArray<ArrayType, DType>> jInputDiff(
      new ZipArray<ArrayType, DType>(env, inputDiff, inputDiffOffset,
                                     ptr->gradInput));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernel(
      new ZipArray<ArrayType, DType>(env, kernel, kernelOffset, ptr->backKernel));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBias(
      new ZipArray<ArrayType, DType>(env, bias, biasOffset, ptr->bias));

  ptr->updateGradInput(jInput->getPtr(), jOutputDiff->getPtr(),
                       jInputDiff->getPtr());
}

template <typename ArrayType, typename DType>
void JNIConvolutionUpdateGradKernel(JNIEnv *env, jclass thisClass,
                                    ArrayType input, jint inputOffset,
                                    ArrayType outputDiff, jint outputDiffOffset,
                                    ArrayType kernelDiff, jint kernelDiffOffset,
                                    ArrayType kernel, jint kernelOffset,
                                    ArrayType bias, jint biasOffset,
                                    long classPtr)
{
  MKLConvolution<DType> *ptr =
      reinterpret_cast<MKLConvolution<DType> *>(classPtr);

  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutputDiff(
      new ZipArray<ArrayType, DType>(env, outputDiff, outputDiffOffset,
                                     ptr->gradOutputK));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernelDiff(
      new ZipArray<ArrayType, DType>(env, kernelDiff, kernelDiffOffset,
                                     ptr->gradKernel));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernel(
      new ZipArray<ArrayType, DType>(env, kernel, kernelOffset, ptr->kernel));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBias(
      new ZipArray<ArrayType, DType>(env, bias, biasOffset, ptr->bias));

  ptr->updateGradKernel(jInput->getPtr(), jOutputDiff->getPtr(),
                        jKernelDiff->getPtr());
}

template <typename ArrayType, typename DType>
void JNIConvolutionUpdateGradBias(JNIEnv *env, jclass thisClass,
                                  ArrayType input, jint inputOffset,
                                  ArrayType outputDiff, jint outputDiffOffset,
                                  ArrayType biasDiff, jint biasDiffOffset,
                                  ArrayType kernel, jint kernelOffset,
                                  ArrayType bias, jint biasOffset,
                                  long classPtr)
{
  MKLConvolution<DType> *ptr =
      reinterpret_cast<MKLConvolution<DType> *>(classPtr);

  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutputDiff(
      new ZipArray<ArrayType, DType>(env, outputDiff, outputDiffOffset,
                                     ptr->gradOutputB));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBiasDiff(
      new ZipArray<ArrayType, DType>(env, biasDiff, biasDiffOffset,
                                     ptr->gradBias));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernel(
      new ZipArray<ArrayType, DType>(env, kernel, kernelOffset, ptr->kernel));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBias(
      new ZipArray<ArrayType, DType>(env, bias, biasOffset, ptr->bias));

  ptr->updateGradBias(jInput->getPtr(), jOutputDiff->getPtr(),
                      jBiasDiff->getPtr());
}

// Macro
#define ConvolutionInit(DType, JType, JArrayType)                             \
  JNIEXPORT                                                                   \
  jlong JNICALL                                                               \
      Java_com_intel_analytics_sparkdl_mkl_MKL_ConvolutionInit##DType(        \
          JNIEnv *env, jclass thisClass, jint inputNumber, jint inputChannel, \
          jint inputHeight, jint inputWidth, jint kernelNumber,               \
          jint kernelChannel, jint kernelHeight, jint kernelWidth,            \
          jint strideHeight, jint strideWidth, jint padHeight, jint padWidth, \
          jint dimension, jint groups, jstring name)                                        \
  {                                                                           \
    return JNIConvolutionInit<JArrayType, JType>(                             \
        env, thisClass, inputNumber, inputChannel, inputHeight, inputWidth,   \
        kernelNumber, kernelChannel, kernelHeight, kernelWidth, strideHeight, \
        strideWidth, padHeight, padWidth, dimension, groups, name);                 \
  }

#define ConvolutionForward(DType, JType, JArrayType)                          \
  JNIEXPORT                                                                   \
  void JNICALL                                                                \
      Java_com_intel_analytics_sparkdl_mkl_MKL_ConvolutionForward##DType(     \
          JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,  \
          JArrayType output, jint outputOffset, JArrayType kernel,            \
          jint kernelOffset, JArrayType bias, jint biasOffset, long classPtr) \
  {                                                                           \
    JNIConvolutionUpdateOutput<JArrayType, JType>(                            \
        env, thisClass, input, inputOffset, output, outputOffset, kernel,     \
        kernelOffset, bias, biasOffset, classPtr);                            \
  }

#define ConvolutionBackwardData(DType, JType, JArrayType)                      \
  JNIEXPORT                                                                    \
  void JNICALL                                                                 \
      Java_com_intel_analytics_sparkdl_mkl_MKL_ConvolutionBackwardData##DType( \
          JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,   \
          JArrayType outputDiff, jint outputDiffOffset, JArrayType inputDiff,  \
          jint inputDiffOffset, JArrayType kernel, jint kernelOffset,          \
          JArrayType bias, jint biasOffset, long classPtr)                     \
  {                                                                            \
    JNIConvolutionUpdateGradInput<JArrayType, JType>(                          \
        env, thisClass, input, inputOffset, outputDiff, outputDiffOffset,      \
        inputDiff, inputDiffOffset, kernel, kernelOffset, bias, biasOffset,    \
        classPtr);                                                             \
  }

#define ConvolutionBackwardKernel(DType, JType, JArrayType)                      \
  JNIEXPORT                                                                      \
  void JNICALL                                                                   \
      Java_com_intel_analytics_sparkdl_mkl_MKL_ConvolutionBackwardKernel##DType( \
          JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,     \
          JArrayType outputDiff, jint outputDiffOffset, JArrayType kernelDiff,   \
          jint kernelDiffOffset, JArrayType kernel, jint kernelOffset,           \
          JArrayType bias, jint biasOffset, long classPtr)                       \
  {                                                                              \
    JNIConvolutionUpdateGradKernel<JArrayType, JType>(                           \
        env, thisClass, input, inputOffset, outputDiff, outputDiffOffset,        \
        kernelDiff, kernelDiffOffset, kernel, kernelOffset, bias, biasOffset,    \
        classPtr);                                                               \
  }

#define ConvolutionBackwardBias(DType, JType, JArrayType)                      \
  JNIEXPORT                                                                    \
  void JNICALL                                                                 \
      Java_com_intel_analytics_sparkdl_mkl_MKL_ConvolutionBackwardBias##DType( \
          JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,   \
          JArrayType outputDiff, jint outputDiffOffset, JArrayType biasDiff,   \
          jint biasDiffOffset, JArrayType kernel, jint kernelOffset,           \
          JArrayType bias, jint biasOffset, long classPtr)                     \
  {                                                                            \
    JNIConvolutionUpdateGradBias<JArrayType, JType>(                           \
        env, thisClass, input, inputOffset, outputDiff, outputDiffOffset,      \
        biasDiff, biasDiffOffset, kernel, kernelOffset, bias, biasOffset,      \
        classPtr);                                                             \
  }

#ifdef __cplusplus
extern "C" {
#endif

// double
ConvolutionInit(Double, jdouble, jdoubleArray);
ConvolutionForward(Double, jdouble, jdoubleArray);
ConvolutionBackwardData(Double, jdouble, jdoubleArray);
ConvolutionBackwardKernel(Double, jdouble, jdoubleArray);
ConvolutionBackwardBias(Double, jdouble, jdoubleArray);

// float
ConvolutionInit(Float, jfloat, jfloatArray);
ConvolutionForward(Float, jfloat, jfloatArray);
ConvolutionBackwardData(Float, jfloat, jfloatArray);
ConvolutionBackwardKernel(Float, jfloat, jfloatArray);
ConvolutionBackwardBias(Float, jfloat, jfloatArray);

#ifdef __cplusplus
}
#endif

#if 0
int main(void)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  MKLConvolution<float> *conv = new MKLConvolution<float>();
  conv->init(32, 64, 56, 56, 192, 64, 3, 3, 1, 1, 1, 1, 4, 1);
  float *input = new float[32 * 64 * 56 * 56];
  int oW = (56 + 2 * 1 - 3) / 1 + 1;
  int oH = (56 + 2 * 1 - 3) / 1 + 1;
  float *output = new float[32 * 192 * oW * oH];
  // std::fill_n(input, 32 * 64 * 56 * 56, 0.1);
  // std::fill_n(output, 32 * 192 * oW * oH, 0.1);

  conv->input->setUsrData(input);
  conv->output->setUsrData(output);

  float *kernel = new float[32 * 192 * 3 * 3 * 2];
  float *bias = new float[192];

  // std::fill_n(kernel, 64 * 3 * 3, 0.1);
  // std::fill_n(bias, 64, 0.1);

  conv->kernel->setUsrData(kernel);
  conv->bias->setUsrData(bias);

  float *gradInput = new float[32 * 64 * 56 * 56];
  float *gradOutput = new float[32 * 192 * oW * oH];

  conv->gradInput->setUsrData(gradInput);
  conv->gradOutput->setUsrData(gradOutput);

  // std::fill_n(gradOutput, 32 * 192 * oW * oH, 0.1);

  float *gradKernel = new float[32 * 192 * 3 * 3 * 2];
  float *gradBias = new float[192];

  conv->gradKernel->setUsrData(gradKernel);
  conv->gradBias->setUsrData(gradBias);

  for (int i = 0; i < 10; i++) {
    conv->updateOutput(input, output);
    conv->updateGradInput(input, gradOutput, gradInput);
    conv->updateGradKernel(input, gradOutput, gradKernel);
    conv->updateGradBias(input, gradOutput, gradBias);
  }
  
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < 20; i++) {
    conv->updateOutput(input, output);
    conv->updateGradInput(input, gradOutput, gradInput);
    conv->updateGradKernel(input, gradOutput, gradKernel);
    conv->updateGradBias(input, gradOutput, gradBias);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  LOG(DBG) << "costs " << (end.tv_sec - start.tv_sec) * 1000 +
    (double)(end.tv_nsec - start.tv_nsec) / 1000000;

  return 0;
}
#endif
