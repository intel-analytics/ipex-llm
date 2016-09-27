#include <stdio.h>
#include <vector>

#include "debug.h"
#include "layer.h"
#include "memory.h"
#include "utils.h"

using namespace std;

template <typename DType>
class MKLSum : public MKLLayer<DType>
{
 public:
  MKLSum();
  ~MKLSum();

  void init(int numSums, int dimension, int *size);

  void updateOutput(DType **input, DType *output);
  void updateGradInput(DType **gradInput, DType *gradOutput);

  // attention, we will override the four variables of MKLLayer
  vector<shared_ptr<MKLData<DType>>> input;

 private:
  void firstPass();
  void preExecute(DType *input);

  int numSums;  // number of concats
  DType *coefficients;
};

template <typename DType>
MKLSum<DType>::MKLSum() : numSums(0)
{
  // TODO
}

template <typename DType>
MKLSum<DType>::~MKLSum()
{
  // TODO
}

template <typename DType>
void MKLSum<DType>::init(int numSums, int dimension, int *size)
{
  this->numSums      = numSums;
  this->dimension    = dimension;
  this->coefficients = new DType[numSums];

  size_t inputSize[dimension];
  size_t inputStrides[dimension];
  size_t outputSize[dimension];
  size_t outputStrides[dimension];

  int offset = 0;

  for (int i = 0; i < numSums; i++) {
    input.push_back(shared_ptr<MKLData<DType>>(new MKLData<DType>));

    // set the size.
    // the size of every channel should be gaved in size.
    // the dimension of every channel should be the same.
    inputStrides[0] = 1;
    inputSize[0]    = size[offset];
    for (int j = 1; j < dimension; j++) {
      inputSize[j]    = size[offset + j];
      inputStrides[j] = inputStrides[j - 1] * inputSize[j - 1];
    }
    offset += dimension;

    this->input[i]->createUsrLayout(dimension, inputSize, inputStrides);
    this->coefficients[i] = 1;
  }

  // TODO check size of all input, they should be the same

  outputStrides[0] = 1;
  outputSize[0]    = inputSize[0];
  for (int i = 1; i < dimension; i++) {
    outputSize[i]    = inputSize[i];
    outputStrides[i] = outputStrides[i - 1] * outputSize[i - 1];
  }

  this->output->createUsrLayout(dimension, outputSize, outputStrides);
}

template <typename DType>
void MKLSum<DType>::firstPass()
{
  dnnLayout_t layout = this->input[0]->getMklLayout();

  dnnError_t status = E_UNIMPLEMENTED;
  status = dnnSumCreate<DType>(&(this->forwardPrim), NULL, numSums, layout,
                               this->coefficients);
  CHECK_EQ(status, E_SUCCESS);

  this->output->createMklLayout(this->forwardPrim, dnnResourceDst);

  for (int i = 0; i < numSums; i++) {
    this->input[i]->createMklLayout(
        this->forwardPrim, (dnnResourceType_t)(dnnResourceMultipleSrc + i));
  }

  this->isFirstPass = false;
}

template <typename DType>
void MKLSum<DType>::updateOutput(DType **input, DType *output)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  if (this->isFirstPass) firstPass();

  for (int i = 0; i < numSums; i++) {
    this->input[i]->setUsrData(input[i]);
    this->input[i]->createConversion();
  }
  this->output->setUsrData(output);
  this->output->createConversion();

  dnnError_t status;
  void *resources[dnnResourceNumber];

  for (int i = 0; i < numSums; i++) {
    resources[dnnResourceMultipleSrc + i] = this->input[i]->getConvertedData();
  }
  resources[dnnResourceDst] = this->output->getData();

  PERFSTART();
  status = dnnExecute<DType>(this->forwardPrim, resources);
  PERFEND("main computing");

  if (!this->output->isUseNext()) this->output->backToUsr();
}

template <typename ArrayType, typename DType>
jlong JNISumInit(JNIEnv *env, jclass thisClass, int numSums, int dimension,
                 jintArray size)
{
  MKLSum<DType> *ptr = new MKLSum<DType>();

  jint *jSize =
      reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(size, 0));
  ptr->init(numSums, dimension, jSize);
  env->ReleasePrimitiveArrayCritical(size, jSize, 0);

  return reinterpret_cast<long>(ptr);
}

template <typename ArrayType, typename DType>
void JNISumUpdateOutput(JNIEnv *env, jclass thisClass, jobjectArray input,
                        jintArray inputOffset, ArrayType output,
                        jint outputOffset, long classPtr)
{
  MKLSum<DType> *ptr = reinterpret_cast<MKLSum<DType> *>(classPtr);

  jint *jInputOffset =
      reinterpret_cast<jint *>(env->GetPrimitiveArrayCritical(inputOffset, 0));

  // TODO we should re-write, this version makes a little complict.
  int len = env->GetArrayLength(input);
  DType *inputArrStart[len];
  DType *inputArr[len];
  ArrayType jInputArr[len];
  for (int i = 0; i < len; i++) {
    jInputArr[i]     = (ArrayType)(env->GetObjectArrayElement(input, i));
    inputArrStart[i] = reinterpret_cast<DType *>(
        env->GetPrimitiveArrayCritical(jInputArr[i], 0));
    inputArr[i] = inputArrStart[i] + jInputOffset[i];
  }

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutput(
      new ZipArray<ArrayType, DType>(env, output, outputOffset, ptr->output));

  ptr->updateOutput(inputArr, jOutput->getPtr());

  for (int i = 0; i < len; i++) {
    env->ReleasePrimitiveArrayCritical(jInputArr[i], inputArrStart[i], 0);
  }

  env->ReleasePrimitiveArrayCritical(inputOffset, jInputOffset, 0);
}

// Macro
#define SumInit(DType, JType, JArrayType)                                    \
  JNIEXPORT                                                                  \
  jlong JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_SumInit##DType(     \
      JNIEnv *env, jclass thisClass, jint numSums, jint dimension,           \
      jintArray size)                                                        \
  {                                                                          \
    return JNISumInit<JArrayType, JType>(env, thisClass, numSums, dimension, \
                                         size);                              \
  }

#define SumForward(DType, JType, JArrayType)                                  \
  JNIEXPORT                                                                   \
  void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_SumForward##DType(    \
      JNIEnv *env, jclass thisClass, jobjectArray input,                      \
      jintArray inputOffset, JArrayType output, jint outputOffset,            \
      long classPtr)                                                          \
  {                                                                           \
    JNISumUpdateOutput<JArrayType, JType>(env, thisClass, input, inputOffset, \
                                          output, outputOffset, classPtr);    \
  }

#ifdef __cplusplus
extern "C" {
#endif

// Double
SumInit(Double, jdouble, jdoubleArray);
SumForward(Double, jdouble, jdoubleArray);

// Float
SumInit(Float, jfloat, jfloatArray);
SumForward(Float, jfloat, jfloatArray);

#ifdef __cplusplus
}
#endif
