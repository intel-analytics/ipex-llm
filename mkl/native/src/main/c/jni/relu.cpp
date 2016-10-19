#include <jni.h>

#include "debug.h"
#include "layer.h"
#include "memory.h"
#include "utils.h"

template <typename DType>
class MKLReLU : public MKLLayer<DType>
{
 public:
  MKLReLU();
  ~MKLReLU();

  void init(size_t inputNumber, size_t inputChannel, size_t inputHeight,
            size_t inputWidth, int dimension, const char *name);

  void updateOutput(DType *input, DType *output);
  void updateGradInput(DType *input, DType *gradOutput, DType *gradInput);

 private:
  // this method is not the same as createMklLayout in MKLMemory
  void firstPass();
  void preExecute(DType *input);

  size_t inputSize[4];
  size_t inputStrides[4];

  size_t outputSize[4];
  size_t outputStrides[4];

  DType nagtiveSlope;
};

template <typename DType>
MKLReLU<DType>::MKLReLU()
{
  nagtiveSlope = static_cast<DType>(0.0);
}

template <typename DType>
MKLReLU<DType>::~MKLReLU()
{
}

template <typename DType>
void MKLReLU<DType>::init(size_t inputNumber, size_t inputChannel,
                          size_t inputHeight, size_t inputWidth, int dimension,
                          const char *name)
{
  this->dimension = dimension;
  this->name.assign(name);

  inputSize[0] = inputWidth;
  inputSize[1] = inputHeight;
  inputSize[2] = inputChannel;
  inputSize[3] = inputNumber;

  inputStrides[0] = 1;
  for (int i        = 1; i < 4; i++)
    inputStrides[i] = inputStrides[i - 1] * inputSize[i - 1];

  // the output channel is as same as the number of kernel.
  // and the output number must be as same as the number of input too.
  outputSize[0] = inputWidth;
  outputSize[1] = inputHeight;
  outputSize[2] = inputChannel;
  outputSize[3] = inputNumber;

  outputStrides[0] = 1;
  for (int i         = 1; i < 4; i++)
    outputStrides[i] = outputStrides[i - 1] * outputSize[i - 1];

  // create usr layout
  this->input->createUsrLayout(dimension, inputSize, inputStrides);
  this->output->createUsrLayout(dimension, outputSize, outputStrides);

  this->gradInput->createUsrLayout(dimension, inputSize, inputStrides);
  this->gradOutput->createUsrLayout(dimension, outputSize, outputStrides);
}

template <typename DType>
void MKLReLU<DType>::firstPass()
{
  dnnError_t status = E_UNIMPLEMENTED;
  dnnLayout_t layout = NULL;

  if (this->input->isUsePrev()) {
    layout = this->input->layoutPrev;
  }
  if (!layout) {
    status =
      dnnLayoutCreate<DType>(&layout, this->dimension, inputSize, inputStrides);
    CHECK_EQ(status, E_SUCCESS);
  } 

  // forward
  status = dnnReLUCreateForward<DType>(&(this->forwardPrim), NULL, layout,
                                       nagtiveSlope);
  CHECK_EQ(status, E_SUCCESS);

  this->input->createMklLayout(this->forwardPrim, dnnResourceSrc);
  this->output->createMklLayout(this->forwardPrim, dnnResourceDst);

  // backward data
  // the input layout is as same as input diff layout
  status = dnnReLUCreateBackward<DType>(&(this->backwardPrim), NULL, layout,
                                        layout, nagtiveSlope);
  CHECK_EQ(status, E_SUCCESS);

  this->gradOutput->createMklLayout(this->backwardPrim, dnnResourceDiffDst);
  this->gradInput->createMklLayout(this->backwardPrim, dnnResourceDiffSrc);

  if (! this->input->isUsePrev()) {
    dnnLayoutDelete<DType>(layout);
  }

  // we create the layout only at the first time
  this->isFirstPass = false;
}

template <typename DType>
void MKLReLU<DType>::preExecute(DType *input)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  this->input->createConversion();
}

template <typename DType>
void MKLReLU<DType>::updateOutput(DType *input, DType *output)
{
  if (this->isFirstPass) firstPass();

  // Because the address will change every time, so we need create conversion
  // every forward/backward.
  // TODO Should we set the kernel and bias address every time?
  preExecute(input);
  this->output->createConversion();

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->input->getUsrData()),
                   this->inputSize[3], this->inputSize[2], this->inputSize[1],
                   this->inputSize[0], "Forward input");
#endif

  dnnError_t status;
  void *resources[dnnResourceNumber];

  resources[dnnResourceSrc] = this->input->getConvertedData();
  resources[dnnResourceDst] = this->output->getData();

  PERFSTART();
  status = dnnExecute<DType>(this->forwardPrim, resources);
  PERFEND("main computing");
  CHECK_EQ(status, E_SUCCESS);

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
void MKLReLU<DType>::updateGradInput(DType *input, DType *gradOutput,
                                     DType *gradInput)
{
  dnnError_t status;
  void *resources[dnnResourceNumber];

  preExecute(input);

  this->gradOutput->createConversion();
  this->gradInput->createConversion();

  resources[dnnResourceDiffDst] = this->gradOutput->getConvertedData();
  resources[dnnResourceDiffSrc] = this->gradInput->getData();
  resources[dnnResourceSrc]     = this->input->getConvertedData();

  // 4. main computing parts.
  PERFSTART();
  status = dnnExecute<DType>(this->backwardPrim, resources);
  CHECK_EQ(status, E_SUCCESS);
  PERFEND("main computing");

  if (!this->gradInput->isUsePrev()) {
    this->gradInput->backToUsr();
  }

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->gradInput->getUsrData()),
                   inputSize[3], inputSize[2], inputSize[1], inputSize[0],
                   "backward gradient input");
#endif
}

template <typename ArrayType, typename DType>
jlong JNIReLUInit(JNIEnv *env, jclass thisClass, jint inputNumber,
                  jint inputChannel, jint inputHeight, jint inputWidth,
                  jint dimension, jstring name)
{
  const char *jName = env->GetStringUTFChars(name, NULL);
  MKLReLU<DType> *ptr = new MKLReLU<DType>();
  ptr->init(inputNumber, inputChannel, inputHeight, inputWidth, dimension, jName);

  return reinterpret_cast<long>(ptr);
}

template <typename ArrayType, typename DType>
void JNIReLUUpdateOutput(JNIEnv *env, jclass thisClass, ArrayType input,
                         jint inputOffset, ArrayType output, jint outputOffset,
                         long classPtr)
{
  MKLReLU<DType> *ptr = reinterpret_cast<MKLReLU<DType> *>(classPtr);

  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutput(
      new ZipArray<ArrayType, DType>(env, output, outputOffset, ptr->output));

  ptr->updateOutput(jInput->getPtr(), jOutput->getPtr());
}

template <typename ArrayType, typename DType>
void JNIReLUUpdateGradInput(JNIEnv *env, jclass thisClass, ArrayType input,
                            jint inputOffset, ArrayType outputDiff,
                            jint outputDiffOffset, ArrayType inputDiff,
                            jint inputDiffOffset, long classPtr)
{
  MKLReLU<DType> *ptr = reinterpret_cast<MKLReLU<DType> *>(classPtr);
  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutputDiff(
      new ZipArray<ArrayType, DType>(env, outputDiff, outputDiffOffset,
                                     ptr->gradOutput));

  std::shared_ptr<ZipArray<ArrayType, DType>> jInputDiff(
      new ZipArray<ArrayType, DType>(env, inputDiff, inputDiffOffset,
                                     ptr->gradInput));

  ptr->updateGradInput(jInput->getPtr(), jOutputDiff->getPtr(),
                       jInputDiff->getPtr());
}

// Macro
#define ReLUInit(DType, JType, JArrayType)                                \
  JNIEXPORT                                                               \
  jlong JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_ReLUInit##DType( \
      JNIEnv *env, jclass thisClass, jint inputNumber, jint inputChannel, \
      jint inputHeight, jint inputWidth, jint dimension, jstring name)                  \
  {                                                                       \
    return JNIReLUInit<JArrayType, JType>(env, thisClass, inputNumber,    \
                                          inputChannel, inputHeight,      \
                                          inputWidth, dimension, name);         \
  }

#define ReLUForward(DType, JType, JArrayType)                                  \
  JNIEXPORT                                                                    \
  void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_ReLUForward##DType(    \
      JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,       \
      JArrayType output, jint outputOffset, long classPtr)                     \
  {                                                                            \
    JNIReLUUpdateOutput<JArrayType, JType>(env, thisClass, input, inputOffset, \
                                           output, outputOffset, classPtr);    \
  }

#define ReLUBackward(DType, JType, JArrayType)                               \
  JNIEXPORT                                                                  \
  void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_ReLUBackward##DType( \
      JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,     \
      JArrayType outputDiff, jint outputDiffOffset, JArrayType inputDiff,    \
      jint inputDiffOffset, long classPtr)                                   \
  {                                                                          \
    JNIReLUUpdateGradInput<JArrayType, JType>(                               \
        env, thisClass, input, inputOffset, outputDiff, outputDiffOffset,    \
        inputDiff, inputDiffOffset, classPtr);                               \
  }

#ifdef __cplusplus
extern "C" {
#endif

// double
ReLUInit(Double, jdouble, jdoubleArray);
ReLUForward(Double, jdouble, jdoubleArray);
ReLUBackward(Double, jdouble, jdoubleArray);

// float
ReLUInit(Float, jfloat, jfloatArray);
ReLUForward(Float, jfloat, jfloatArray);
ReLUBackward(Float, jfloat, jfloatArray);

#ifdef __cplusplus
}
#endif
