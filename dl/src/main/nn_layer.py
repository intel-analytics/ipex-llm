class Abs(Model):
    '''
    >>> abs = Abs()
    creating: createAbs
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Abs, self).__init__(None, bigdl_type)

class AbsCriterion(Model):
    '''
    >>> absCriterion = AbsCriterion(Boolean)
    creating: createAbsCriterion
    '''

    def __init__(self,
                size_average=True,
                bigdl_type="float"):
        super(AbsCriterion, self).__init__(None, bigdl_type,
                                           size_average)

class Add(Model):
    '''
    >>> add = Add(Int)
    creating: createAdd
    '''

    def __init__(self,
                input_size,
                bigdl_type="float"):
        super(Add, self).__init__(None, bigdl_type,
                                  input_size)

class AddConstant(Model):
    '''
    >>> addConstant = AddConstant(T, Boolean)
    creating: createAddConstant
    '''

    def __init__(self,
                constant_scalar,
                inplace=False,
                bigdl_type="float"):
        super(AddConstant, self).__init__(None, bigdl_type,
                                          constant_scalar,
                                          inplace)

class BCECriterion(Model):
    '''
    >>> bCECriterion = BCECriterion(Tensor, Boolean)
    creating: createBCECriterion
    '''

    def __init__(self,
                weights,
                size_average=True,
                bigdl_type="float"):
        super(BCECriterion, self).__init__(None, bigdl_type,
                                           weights,
                                           size_average)

class BatchNormalization(Model):
    '''
    >>> batchNormalization = BatchNormalization(Int, Double, Double, Boolean)
    creating: createBatchNormalization
    '''

    def __init__(self,
                n_output,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                bigdl_type="float"):
        super(BatchNormalization, self).__init__(None, bigdl_type,
                                                 n_output,
                                                 eps,
                                                 momentum,
                                                 affine)

class Bilinear(Model):
    '''
    >>> bilinear = Bilinear(Int, Int, Int, Boolean)
    creating: createBilinear
    '''

    def __init__(self,
                input_size1,
                input_size2,
                output_size,
                bias_res=True,
                bigdl_type="float"):
        super(Bilinear, self).__init__(None, bigdl_type,
                                       input_size1,
                                       input_size2,
                                       output_size,
                                       bias_res)

class Bottle(Model):
    '''
    >>> bottle = Bottle(Module, Int, Int)
    creating: createBottle
    '''

    def __init__(self,
                module,
                n_input_dim=2,
                n_output_dim1=Int.MaxValue,
                bigdl_type="float"):
        super(Bottle, self).__init__(None, bigdl_type,
                                     module,
                                     n_input_dim,
                                     n_output_dim1)

class CAdd(Model):
    '''
    >>> cAdd = CAdd(Array)
    creating: createCAdd
    '''

    def __init__(self,
                size,
                bigdl_type="float"):
        super(CAdd, self).__init__(None, bigdl_type,
                                   size)

class CAddTable(Model):
    '''
    >>> cAddTable = CAddTable(Boolean)
    creating: createCAddTable
    '''

    def __init__(self,
                inplace=False,
                bigdl_type="float"):
        super(CAddTable, self).__init__(None, bigdl_type,
                                        inplace)

class CDivTable(Model):
    '''
    >>> cDivTable = CDivTable()
    creating: createCDivTable
    '''

    def __init__(self,
                bigdl_type="float"):
        super(CDivTable, self).__init__(None, bigdl_type)

class CMaxTable(Model):
    '''
    >>> cMaxTable = CMaxTable()
    creating: createCMaxTable
    '''

    def __init__(self,
                bigdl_type="float"):
        super(CMaxTable, self).__init__(None, bigdl_type)

class CMinTable(Model):
    '''
    >>> cMinTable = CMinTable()
    creating: createCMinTable
    '''

    def __init__(self,
                bigdl_type="float"):
        super(CMinTable, self).__init__(None, bigdl_type)

class CMul(Model):
    '''
    >>> cMul = CMul(Array)
    creating: createCMul
    '''

    def __init__(self,
                size,
                bigdl_type="float"):
        super(CMul, self).__init__(None, bigdl_type,
                                   size)

class CMulTable(Model):
    '''
    >>> cMulTable = CMulTable()
    creating: createCMulTable
    '''

    def __init__(self,
                bigdl_type="float"):
        super(CMulTable, self).__init__(None, bigdl_type)

class CSubTable(Model):
    '''
    >>> cSubTable = CSubTable()
    creating: createCSubTable
    '''

    def __init__(self,
                bigdl_type="float"):
        super(CSubTable, self).__init__(None, bigdl_type)

class Clamp(Model):
    '''
    >>> clamp = Clamp(Int, Int)
    creating: createClamp
    '''

    def __init__(self,
                min,
                max,
                bigdl_type="float"):
        super(Clamp, self).__init__(None, bigdl_type,
                                    min,
                                    max)

class ClassNLLCriterion(Model):
    '''
    >>> classNLLCriterion = ClassNLLCriterion(Tensor, Boolean)
    creating: createClassNLLCriterion
    '''

    def __init__(self,
                weights,
                size_average=True,
                bigdl_type="float"):
        super(ClassNLLCriterion, self).__init__(None, bigdl_type,
                                                weights,
                                                size_average)

class ClassSimplexCriterion(Model):
    '''
    >>> classSimplexCriterion = ClassSimplexCriterion(Int)
    creating: createClassSimplexCriterion
    '''

    def __init__(self,
                n_classes,
                bigdl_type="float"):
        super(ClassSimplexCriterion, self).__init__(None, bigdl_type,
                                                    n_classes)

class Concat(Model):
    '''
    >>> concat = Concat(Int)
    creating: createConcat
    '''

    def __init__(self,
                dimension,
                bigdl_type="float"):
        super(Concat, self).__init__(None, bigdl_type,
                                     dimension)

class ConcatTable(Model):
    '''
    >>> concatTable = ConcatTable()
    creating: createConcatTable
    '''

    def __init__(self,
                bigdl_type="float"):
        super(ConcatTable, self).__init__(None, bigdl_type)

class Contiguous(Model):
    '''
    >>> contiguous = Contiguous()
    creating: createContiguous
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Contiguous, self).__init__(None, bigdl_type)

class Copy(Model):
    '''
    >>> copy = Copy()
    creating: createCopy
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Copy, self).__init__(None, bigdl_type)

class Cosine(Model):
    '''
    >>> cosine = Cosine()
    creating: createCosine
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Cosine, self).__init__(None, bigdl_type)

class CosineDistance(Model):
    '''
    >>> cosineDistance = CosineDistance()
    creating: createCosineDistance
    '''

    def __init__(self,
                bigdl_type="float"):
        super(CosineDistance, self).__init__(None, bigdl_type)

class CosineEmbeddingCriterion(Model):
    '''
    >>> cosineEmbeddingCriterion = CosineEmbeddingCriterion(Double, Boolean)
    creating: createCosineEmbeddingCriterion
    '''

    def __init__(self,
                margin=0.0,
                size_average=True,
                bigdl_type="float"):
        super(CosineEmbeddingCriterion, self).__init__(None, bigdl_type,
                                                       margin,
                                                       size_average)

class CriterionTable(Model):
    '''
    >>> criterionTable = CriterionTable(TensorCriterion)
    creating: createCriterionTable
    '''

    def __init__(self,
                criterion,
                bigdl_type="float"):
        super(CriterionTable, self).__init__(None, bigdl_type,
                                             criterion)

class CrossEntropyCriterion(Model):
    '''
    >>> crossEntropyCriterion = CrossEntropyCriterion(Tensor)
    creating: createCrossEntropyCriterion
    '''

    def __init__(self,
                weights,
                bigdl_type="float"):
        super(CrossEntropyCriterion, self).__init__(None, bigdl_type,
                                                    weights)

class DistKLDivCriterion(Model):
    '''
    >>> distKLDivCriterion = DistKLDivCriterion(Boolean)
    creating: createDistKLDivCriterion
    '''

    def __init__(self,
                size_average=True,
                bigdl_type="float"):
        super(DistKLDivCriterion, self).__init__(None, bigdl_type,
                                                 size_average)

class DotProduct(Model):
    '''
    >>> dotProduct = DotProduct()
    creating: createDotProduct
    '''

    def __init__(self,
                bigdl_type="float"):
        super(DotProduct, self).__init__(None, bigdl_type)

class Dropout(Model):
    '''
    >>> dropout = Dropout(Double, Boolean, Boolean)
    creating: createDropout
    '''

    def __init__(self,
                init_p=0.5,
                inplace=False,
                scale=True,
                bigdl_type="float"):
        super(Dropout, self).__init__(None, bigdl_type,
                                      init_p,
                                      inplace,
                                      scale)

class ELU(Model):
    '''
    >>> eLU = ELU(Double, Boolean)
    creating: createELU
    '''

    def __init__(self,
                alpha=1.0,
                inplace=False,
                bigdl_type="float"):
        super(ELU, self).__init__(None, bigdl_type,
                                  alpha,
                                  inplace)

class Echo(Model):
    '''
    >>> echo = Echo()
    creating: createEcho
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Echo, self).__init__(None, bigdl_type)

class ErrorInfo(Model):
    '''
    >>> errorInfo = ErrorInfo()
    creating: createErrorInfo
    '''

    def __init__(self,
                bigdl_type="float"):
        super(ErrorInfo, self).__init__(None, bigdl_type)

class Euclidean(Model):
    '''
    >>> euclidean = Euclidean(Int, Int, Boolean)
    creating: createEuclidean
    '''

    def __init__(self,
                input_size,
                output_size,
                fast_backward=True,
                bigdl_type="float"):
        super(Euclidean, self).__init__(None, bigdl_type,
                                        input_size,
                                        output_size,
                                        fast_backward)

class Exp(Model):
    '''
    >>> exp = Exp()
    creating: createExp
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Exp, self).__init__(None, bigdl_type)

class FlattenTable(Model):
    '''
    >>> flattenTable = FlattenTable()
    creating: createFlattenTable
    '''

    def __init__(self,
                bigdl_type="float"):
        super(FlattenTable, self).__init__(None, bigdl_type)

class GradientReversal(Model):
    '''
    >>> gradientReversal = GradientReversal(Double)
    creating: createGradientReversal
    '''

    def __init__(self,
                lambda=1,
                bigdl_type="float"):
        super(GradientReversal, self).__init__(None, bigdl_type,
                                               lambda)

class HardShrink(Model):
    '''
    >>> hardShrink = HardShrink(Double)
    creating: createHardShrink
    '''

    def __init__(self,
                lambda=0.5,
                bigdl_type="float"):
        super(HardShrink, self).__init__(None, bigdl_type,
                                         lambda)

class HardTanh(Model):
    '''
    >>> hardTanh = HardTanh(Double, Double, Boolean)
    creating: createHardTanh
    '''

    def __init__(self,
                min_value=-1,
                max_value=1,
                inplace=False,
                bigdl_type="float"):
        super(HardTanh, self).__init__(None, bigdl_type,
                                       min_value,
                                       max_value,
                                       inplace)

class HingeEmbeddingCriterion(Model):
    '''
    >>> hingeEmbeddingCriterion = HingeEmbeddingCriterion(Double, Boolean)
    creating: createHingeEmbeddingCriterion
    '''

    def __init__(self,
                margin=1,
                size_average=True,
                bigdl_type="float"):
        super(HingeEmbeddingCriterion, self).__init__(None, bigdl_type,
                                                      margin,
                                                      size_average)

class Identity(Model):
    '''
    >>> identity = Identity()
    creating: createIdentity
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Identity, self).__init__(None, bigdl_type)

class Index(Model):
    '''
    >>> index = Index(Int)
    creating: createIndex
    '''

    def __init__(self,
                dimension,
                bigdl_type="float"):
        super(Index, self).__init__(None, bigdl_type,
                                    dimension)

class InferReshape(Model):
    '''
    >>> inferReshape = InferReshape(Array, Boolean)
    creating: createInferReshape
    '''

    def __init__(self,
                size,
                batch_mode=False,
                bigdl_type="float"):
        super(InferReshape, self).__init__(None, bigdl_type,
                                           size,
                                           batch_mode)

class JoinTable(Model):
    '''
    >>> joinTable = JoinTable(Int, Int)
    creating: createJoinTable
    '''

    def __init__(self,
                dimension,
                n_input_dims,
                bigdl_type="float"):
        super(JoinTable, self).__init__(None, bigdl_type,
                                        dimension,
                                        n_input_dims)

class L1Cost(Model):
    '''
    >>> l1Cost = L1Cost()
    creating: createL1Cost
    '''

    def __init__(self,
                bigdl_type="float"):
        super(L1Cost, self).__init__(None, bigdl_type)

class L1HingeEmbeddingCriterion(Model):
    '''
    >>> l1HingeEmbeddingCriterion = L1HingeEmbeddingCriterion(Double)
    creating: createL1HingeEmbeddingCriterion
    '''

    def __init__(self,
                margin=1,
                bigdl_type="float"):
        super(L1HingeEmbeddingCriterion, self).__init__(None, bigdl_type,
                                                        margin)

class L1Penalty(Model):
    '''
    >>> l1Penalty = L1Penalty(Int, Boolean, Boolean)
    creating: createL1Penalty
    '''

    def __init__(self,
                l1weight,
                size_average=False,
                provide_output=True,
                bigdl_type="float"):
        super(L1Penalty, self).__init__(None, bigdl_type,
                                        l1weight,
                                        size_average,
                                        provide_output)

class LeakyReLU(Model):
    '''
    >>> leakyReLU = LeakyReLU(Double, Boolean)
    creating: createLeakyReLU
    '''

    def __init__(self,
                negval=0.01,
                inplace=False,
                bigdl_type="float"):
        super(LeakyReLU, self).__init__(None, bigdl_type,
                                        negval,
                                        inplace)

class Linear(Model):
    '''
    >>> linear = Linear(Int, Int, InitializationMethod, Boolean)
    creating: createLinear
    '''

    def __init__(self,
                input_size,
                output_size,
                init_method=Default,
                with_bias=True,
                bigdl_type="float"):
        super(Linear, self).__init__(None, bigdl_type,
                                     input_size,
                                     output_size,
                                     init_method,
                                     with_bias)

class Log(Model):
    '''
    >>> log = Log()
    creating: createLog
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Log, self).__init__(None, bigdl_type)

class LogSigmoid(Model):
    '''
    >>> logSigmoid = LogSigmoid()
    creating: createLogSigmoid
    '''

    def __init__(self,
                bigdl_type="float"):
        super(LogSigmoid, self).__init__(None, bigdl_type)

class LogSoftMax(Model):
    '''
    >>> logSoftMax = LogSoftMax()
    creating: createLogSoftMax
    '''

    def __init__(self,
                bigdl_type="float"):
        super(LogSoftMax, self).__init__(None, bigdl_type)

class LookupTable(Model):
    '''
    >>> lookupTable = LookupTable(Int, Int, Double, Double, Double, Boolean)
    creating: createLookupTable
    '''

    def __init__(self,
                n_index,
                n_output,
                padding_value=0,
                max_norm=Double.MaxValue,
                norm_type=2.0,
                should_scale_grad_by_freq=False,
                bigdl_type="float"):
        super(LookupTable, self).__init__(None, bigdl_type,
                                          n_index,
                                          n_output,
                                          padding_value,
                                          max_norm,
                                          norm_type,
                                          should_scale_grad_by_freq)

class MM(Model):
    '''
    >>> mM = MM(Boolean, Boolean)
    creating: createMM
    '''

    def __init__(self,
                trans_a=False,
                trans_b=False,
                bigdl_type="float"):
        super(MM, self).__init__(None, bigdl_type,
                                 trans_a,
                                 trans_b)

class MSECriterion(Model):
    '''
    >>> mSECriterion = MSECriterion()
    creating: createMSECriterion
    '''

    def __init__(self,
                bigdl_type="float"):
        super(MSECriterion, self).__init__(None, bigdl_type)

class MV(Model):
    '''
    >>> mV = MV(Boolean)
    creating: createMV
    '''

    def __init__(self,
                trans=False,
                bigdl_type="float"):
        super(MV, self).__init__(None, bigdl_type,
                                 trans)

class MapTable(Model):
    '''
    >>> mapTable = MapTable(AbstractModule)
    creating: createMapTable
    '''

    def __init__(self,
                module,
                bigdl_type="float"):
        super(MapTable, self).__init__(None, bigdl_type,
                                       module)

class MarginCriterion(Model):
    '''
    >>> marginCriterion = MarginCriterion(Double, Boolean)
    creating: createMarginCriterion
    '''

    def __init__(self,
                margin=1.0,
                size_average=True,
                bigdl_type="float"):
        super(MarginCriterion, self).__init__(None, bigdl_type,
                                              margin,
                                              size_average)

class MarginRankingCriterion(Model):
    '''
    >>> marginRankingCriterion = MarginRankingCriterion(Double, Boolean)
    creating: createMarginRankingCriterion
    '''

    def __init__(self,
                margin=1.0,
                size_average=True,
                bigdl_type="float"):
        super(MarginRankingCriterion, self).__init__(None, bigdl_type,
                                                     margin,
                                                     size_average)

class MaskedSelect(Model):
    '''
    >>> maskedSelect = MaskedSelect()
    creating: createMaskedSelect
    '''

    def __init__(self,
                bigdl_type="float"):
        super(MaskedSelect, self).__init__(None, bigdl_type)

class Max(Model):
    '''
    >>> max = Max(TensorNumeric)
    creating: createMax
    '''

    def __init__(self,
                ev,
                bigdl_type="float"):
        super(Max, self).__init__(None, bigdl_type,
                                  ev)

class Mean(Model):
    '''
    >>> mean = Mean(Int, Int)
    creating: createMean
    '''

    def __init__(self,
                dimension=1,
                n_input_dims=-1,
                bigdl_type="float"):
        super(Mean, self).__init__(None, bigdl_type,
                                   dimension,
                                   n_input_dims)

class Min(Model):
    '''
    >>> min = Min(Int)
    creating: createMin
    '''

    def __init__(self,
                num_input_dims=Int.MinValue,
                bigdl_type="float"):
        super(Min, self).__init__(None, bigdl_type,
                                  num_input_dims)

class MixtureTable(Model):
    '''
    >>> mixtureTable = MixtureTable()
    creating: createMixtureTable
    '''

    def __init__(self,
                bigdl_type="float"):
        super(MixtureTable, self).__init__(None, bigdl_type)

class Module(Model):
    '''
    >>> module = Module()
    creating: createModule
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Module, self).__init__(None, bigdl_type)

class Mul(Model):
    '''
    >>> mul = Mul()
    creating: createMul
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Mul, self).__init__(None, bigdl_type)

class MulConstant(Model):
    '''
    >>> mulConstant = MulConstant()
    creating: createMulConstant
    '''

    def __init__(self,
                bigdl_type="float"):
        super(MulConstant, self).__init__(None, bigdl_type)

class MultiCriterion(Model):
    '''
    >>> multiCriterion = MultiCriterion()
    creating: createMultiCriterion
    '''

    def __init__(self,
                bigdl_type="float"):
        super(MultiCriterion, self).__init__(None, bigdl_type)

class MultiLabelMarginCriterion(Model):
    '''
    >>> multiLabelMarginCriterion = MultiLabelMarginCriterion(Boolean)
    creating: createMultiLabelMarginCriterion
    '''

    def __init__(self,
                size_average=True,
                bigdl_type="float"):
        super(MultiLabelMarginCriterion, self).__init__(None, bigdl_type,
                                                        size_average)

class MultiLabelSoftMarginCriterion(Model):
    '''
    >>> multiLabelSoftMarginCriterion = MultiLabelSoftMarginCriterion(Tensor, Boolean)
    creating: createMultiLabelSoftMarginCriterion
    '''

    def __init__(self,
                weights,
                size_average=True,
                bigdl_type="float"):
        super(MultiLabelSoftMarginCriterion, self).__init__(None, bigdl_type,
                                                            weights,
                                                            size_average)

class MultiMarginCriterion(Model):
    '''
    >>> multiMarginCriterion = MultiMarginCriterion(Int, Tensor, Double, Boolean)
    creating: createMultiMarginCriterion
    '''

    def __init__(self,
                p,
                weights=1,
                margin=1.0,
                size_average=True,
                bigdl_type="float"):
        super(MultiMarginCriterion, self).__init__(None, bigdl_type,
                                                   p,
                                                   weights,
                                                   margin,
                                                   size_average)

class NNPrimitive(Model):
    '''
    >>> nNPrimitive = NNPrimitive()
    creating: createNNPrimitive
    '''

    def __init__(self,
                bigdl_type="float"):
        super(NNPrimitive, self).__init__(None, bigdl_type)

class Narrow(Model):
    '''
    >>> narrow = Narrow(Int, Int, Int)
    creating: createNarrow
    '''

    def __init__(self,
                dimension,
                offset,
                length=1,
                bigdl_type="float"):
        super(Narrow, self).__init__(None, bigdl_type,
                                     dimension,
                                     offset,
                                     length)

class NarrowTable(Model):
    '''
    >>> narrowTable = NarrowTable(Int, Int)
    creating: createNarrowTable
    '''

    def __init__(self,
                offset,
                length=1,
                bigdl_type="float"):
        super(NarrowTable, self).__init__(None, bigdl_type,
                                          offset,
                                          length)

class Normalize(Model):
    '''
    >>> normalize = Normalize(Double, Double)
    creating: createNormalize
    '''

    def __init__(self,
                p,
                eps=1e-10,
                bigdl_type="float"):
        super(Normalize, self).__init__(None, bigdl_type,
                                        p,
                                        eps)

class PReLU(Model):
    '''
    >>> pReLU = PReLU(Int)
    creating: createPReLU
    '''

    def __init__(self,
                n_output_plane=0,
                bigdl_type="float"):
        super(PReLU, self).__init__(None, bigdl_type,
                                    n_output_plane)

class Padding(Model):
    '''
    >>> padding = Padding(Int, Int, Int, Double, Int)
    creating: createPadding
    '''

    def __init__(self,
                dim,
                pad,
                n_input_dim,
                value=0.0,
                n_index=1,
                bigdl_type="float"):
        super(Padding, self).__init__(None, bigdl_type,
                                      dim,
                                      pad,
                                      n_input_dim,
                                      value,
                                      n_index)

class PairwiseDistance(Model):
    '''
    >>> pairwiseDistance = PairwiseDistance()
    creating: createPairwiseDistance
    '''

    def __init__(self,
                bigdl_type="float"):
        super(PairwiseDistance, self).__init__(None, bigdl_type)

class ParallelCriterion(Model):
    '''
    >>> parallelCriterion = ParallelCriterion(Boolean)
    creating: createParallelCriterion
    '''

    def __init__(self,
                repeat_target=False,
                bigdl_type="float"):
        super(ParallelCriterion, self).__init__(None, bigdl_type,
                                                repeat_target)

class ParallelTable(Model):
    '''
    >>> parallelTable = ParallelTable()
    creating: createParallelTable
    '''

    def __init__(self,
                bigdl_type="float"):
        super(ParallelTable, self).__init__(None, bigdl_type)

class Power(Model):
    '''
    >>> power = Power(Double)
    creating: createPower
    '''

    def __init__(self,
                power,
                bigdl_type="float"):
        super(Power, self).__init__(None, bigdl_type,
                                    power)

class RnnCell(Model):
    '''
    >>> rnnCell = RnnCell(Int, Int)
    creating: createRnnCell
    '''

    def __init__(self,
                input_size=4,
                hidden_size=3,
                bigdl_type="float"):
        super(RnnCell, self).__init__(None, bigdl_type,
                                      input_size,
                                      hidden_size)

class RReLU(Model):
    '''
    >>> rReLU = RReLU(Double, Double, Boolean)
    creating: createRReLU
    '''

    def __init__(self,
                lower=1.0/8,
                upper=1.0/3,
                inplace=False,
                bigdl_type="float"):
        super(RReLU, self).__init__(None, bigdl_type,
                                    lower,
                                    upper,
                                    inplace)

class ReLU(Model):
    '''
    >>> reLU = ReLU(Boolean)
    creating: createReLU
    '''

    def __init__(self,
                ip=False,
                bigdl_type="float"):
        super(ReLU, self).__init__(None, bigdl_type,
                                   ip)

class ReLU6(Model):
    '''
    >>> reLU6 = ReLU6(Boolean)
    creating: createReLU6
    '''

    def __init__(self,
                inplace=False,
                bigdl_type="float"):
        super(ReLU6, self).__init__(None, bigdl_type,
                                    inplace)

class Recurrent(Model):
    '''
    >>> recurrent = Recurrent(Int, Int)
    creating: createRecurrent
    '''

    def __init__(self,
                hidden_size=3,
                bptt_truncate=2,
                bigdl_type="float"):
        super(Recurrent, self).__init__(None, bigdl_type,
                                        hidden_size,
                                        bptt_truncate)

class Replicate(Model):
    '''
    >>> replicate = Replicate()
    creating: createReplicate
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Replicate, self).__init__(None, bigdl_type)

class Reshape(Model):
    '''
    >>> reshape = Reshape(Array, Option)
    creating: createReshape
    '''

    def __init__(self,
                size,
                batch_mode,
                bigdl_type="float"):
        super(Reshape, self).__init__(None, bigdl_type,
                                      size,
                                      batch_mode)

class RoiPooling(Model):
    '''
    >>> roiPooling = RoiPooling(Int, Int, T)
    creating: createRoiPooling
    '''

    def __init__(self,
                pooled_w,
                pooled_h,
                spatial_scale,
                bigdl_type="float"):
        super(RoiPooling, self).__init__(None, bigdl_type,
                                         pooled_w,
                                         pooled_h,
                                         spatial_scale)

class Scale(Model):
    '''
    >>> scale = Scale(Array)
    creating: createScale
    '''

    def __init__(self,
                size,
                bigdl_type="float"):
        super(Scale, self).__init__(None, bigdl_type,
                                    size)

class Select(Model):
    '''
    >>> select = Select(Int, Int)
    creating: createSelect
    '''

    def __init__(self,
                dimension,
                index,
                bigdl_type="float"):
        super(Select, self).__init__(None, bigdl_type,
                                     dimension,
                                     index)

class Sequential(Model):
    '''
    >>> sequential = Sequential()
    creating: createSequential
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Sequential, self).__init__(None, bigdl_type)

class Sigmoid(Model):
    '''
    >>> sigmoid = Sigmoid()
    creating: createSigmoid
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Sigmoid, self).__init__(None, bigdl_type)

class SmoothL1Criterion(Model):
    '''
    >>> smoothL1Criterion = SmoothL1Criterion(Boolean)
    creating: createSmoothL1Criterion
    '''

    def __init__(self,
                size_average=True,
                bigdl_type="float"):
        super(SmoothL1Criterion, self).__init__(None, bigdl_type,
                                                size_average)

class SmoothL1CriterionWithWeights(Model):
    '''
    >>> smoothL1CriterionWithWeights = SmoothL1CriterionWithWeights(Double, Int)
    creating: createSmoothL1CriterionWithWeights
    '''

    def __init__(self,
                sigma,
                num=0,
                bigdl_type="float"):
        super(SmoothL1CriterionWithWeights, self).__init__(None, bigdl_type,
                                                           sigma,
                                                           num)

class SoftMax(Model):
    '''
    >>> softMax = SoftMax()
    creating: createSoftMax
    '''

    def __init__(self,
                bigdl_type="float"):
        super(SoftMax, self).__init__(None, bigdl_type)

class SoftMin(Model):
    '''
    >>> softMin = SoftMin()
    creating: createSoftMin
    '''

    def __init__(self,
                bigdl_type="float"):
        super(SoftMin, self).__init__(None, bigdl_type)

class SoftPlus(Model):
    '''
    >>> softPlus = SoftPlus(Double)
    creating: createSoftPlus
    '''

    def __init__(self,
                beta=1.0,
                bigdl_type="float"):
        super(SoftPlus, self).__init__(None, bigdl_type,
                                       beta)

class SoftShrink(Model):
    '''
    >>> softShrink = SoftShrink(Double)
    creating: createSoftShrink
    '''

    def __init__(self,
                lambda=0.5,
                bigdl_type="float"):
        super(SoftShrink, self).__init__(None, bigdl_type,
                                         lambda)

class SoftSign(Model):
    '''
    >>> softSign = SoftSign()
    creating: createSoftSign
    '''

    def __init__(self,
                bigdl_type="float"):
        super(SoftSign, self).__init__(None, bigdl_type)

class NormMode(Model):
    '''
    >>> normMode = NormMode(Option, NormMode)
    creating: createNormMode
    '''

    def __init__(self,
                ignore_label,
                normalize_mode=NormMode.VALID,
                bigdl_type="float"):
        super(NormMode, self).__init__(None, bigdl_type,
                                       ignore_label,
                                       normalize_mode)

class SpatialAveragePooling(Model):
    '''
    >>> spatialAveragePooling = SpatialAveragePooling(Int, Int, Int, Int, Int, Int, Boolean, Boolean, Boolean)
    creating: createSpatialAveragePooling
    '''

    def __init__(self,
                kw,
                kh,
                dw=1,
                dh=1,
                pad_w=0,
                pad_h=0,
                ceil_mode=False,
                count_include_pad=True,
                divide=True,
                bigdl_type="float"):
        super(SpatialAveragePooling, self).__init__(None, bigdl_type,
                                                    kw,
                                                    kh,
                                                    dw,
                                                    dh,
                                                    pad_w,
                                                    pad_h,
                                                    ceil_mode,
                                                    count_include_pad,
                                                    divide)

class SpatialBatchNormalization(Model):
    '''
    >>> spatialBatchNormalization = SpatialBatchNormalization(Int, Double, Double, Boolean)
    creating: createSpatialBatchNormalization
    '''

    def __init__(self,
                n_output,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                bigdl_type="float"):
        super(SpatialBatchNormalization, self).__init__(None, bigdl_type,
                                                        n_output,
                                                        eps,
                                                        momentum,
                                                        affine)

class SpatialContrastiveNormalization(Model):
    '''
    >>> spatialContrastiveNormalization = SpatialContrastiveNormalization(Int, Tensor, Double, Double)
    creating: createSpatialContrastiveNormalization
    '''

    def __init__(self,
                n_input_plane,
                kernel=1,
                threshold=1e-4,
                thresval=1e-4,
                bigdl_type="float"):
        super(SpatialContrastiveNormalization, self).__init__(None, bigdl_type,
                                                              n_input_plane,
                                                              kernel,
                                                              threshold,
                                                              thresval)

class SpatialConvolution(Model):
    '''
    >>> spatialConvolution = SpatialConvolution(Int, Int, Int, Int, Int, Int, Int, Int, Int, Boolean, InitializationMethod)
    creating: createSpatialConvolution
    '''

    def __init__(self,
                n_input_plane,
                n_output_plane,
                kernel_w,
                kernel_h,
                stride_w=1,
                stride_h=1,
                pad_w=0,
                pad_h=0,
                n_group=1,
                propagate_back=True,
                init_method=Default,
                bigdl_type="float"):
        super(SpatialConvolution, self).__init__(None, bigdl_type,
                                                 n_input_plane,
                                                 n_output_plane,
                                                 kernel_w,
                                                 kernel_h,
                                                 stride_w,
                                                 stride_h,
                                                 pad_w,
                                                 pad_h,
                                                 n_group,
                                                 propagate_back,
                                                 init_method)

class SpatialConvolutionMap(Model):
    '''
    >>> spatialConvolutionMap = SpatialConvolutionMap(Tensor, Int, Int, Int, Int, Int, Int)
    creating: createSpatialConvolutionMap
    '''

    def __init__(self,
                conn_table,
                kw,
                kh,
                dw=1,
                dh=1,
                pad_w=0,
                pad_h=0,
                bigdl_type="float"):
        super(SpatialConvolutionMap, self).__init__(None, bigdl_type,
                                                    conn_table,
                                                    kw,
                                                    kh,
                                                    dw,
                                                    dh,
                                                    pad_w,
                                                    pad_h)

class SpatialCrossMapLRN(Model):
    '''
    >>> spatialCrossMapLRN = SpatialCrossMapLRN(Int, Double, Double, Double)
    creating: createSpatialCrossMapLRN
    '''

    def __init__(self,
                size=5,
                alpha=1.0,
                beta=0.75,
                k=1.0,
                bigdl_type="float"):
        super(SpatialCrossMapLRN, self).__init__(None, bigdl_type,
                                                 size,
                                                 alpha,
                                                 beta,
                                                 k)

class SpatialDilatedConvolution(Model):
    '''
    >>> spatialDilatedConvolution = SpatialDilatedConvolution(Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, InitializationMethod)
    creating: createSpatialDilatedConvolution
    '''

    def __init__(self,
                n_input_plane,
                n_output_plane,
                kw,
                kh,
                dw=1,
                dh=1,
                pad_w=0,
                pad_h=0,
                dilation_w=1,
                dilation_h=1,
                init_method=Default,
                bigdl_type="float"):
        super(SpatialDilatedConvolution, self).__init__(None, bigdl_type,
                                                        n_input_plane,
                                                        n_output_plane,
                                                        kw,
                                                        kh,
                                                        dw,
                                                        dh,
                                                        pad_w,
                                                        pad_h,
                                                        dilation_w,
                                                        dilation_h,
                                                        init_method)

class SpatialDivisiveNormalization(Model):
    '''
    >>> spatialDivisiveNormalization = SpatialDivisiveNormalization(Int, Tensor, Double, Double)
    creating: createSpatialDivisiveNormalization
    '''

    def __init__(self,
                n_input_plane,
                kernel=1,
                threshold=1e-4,
                thresval=1e-4,
                bigdl_type="float"):
        super(SpatialDivisiveNormalization, self).__init__(None, bigdl_type,
                                                           n_input_plane,
                                                           kernel,
                                                           threshold,
                                                           thresval)

class SpatialFullConvolution(Model):
    '''
    >>> spatialFullConvolution = SpatialFullConvolution(Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Boolean, InitializationMethod)
    creating: createSpatialFullConvolution
    '''

    def __init__(self,
                n_input_plane,
                n_output_plane,
                kw,
                kh,
                dw=1,
                dh=1,
                pad_w=0,
                pad_h=0,
                adj_w=0,
                adj_h=0,
                n_group=1,
                no_bias=False,
                init_method=Default,
                bigdl_type="float"):
        super(SpatialFullConvolution, self).__init__(None, bigdl_type,
                                                     n_input_plane,
                                                     n_output_plane,
                                                     kw,
                                                     kh,
                                                     dw,
                                                     dh,
                                                     pad_w,
                                                     pad_h,
                                                     adj_w,
                                                     adj_h,
                                                     n_group,
                                                     no_bias,
                                                     init_method)

class SpatialMaxPooling(Model):
    '''
    >>> spatialMaxPooling = SpatialMaxPooling(Int, Int, Int, Int, Int, Int)
    creating: createSpatialMaxPooling
    '''

    def __init__(self,
                kw,
                kh,
                dw,
                dh,
                pad_w=0,
                pad_h=0,
                bigdl_type="float"):
        super(SpatialMaxPooling, self).__init__(None, bigdl_type,
                                                kw,
                                                kh,
                                                dw,
                                                dh,
                                                pad_w,
                                                pad_h)

class SpatialShareConvolution(Model):
    '''
    >>> spatialShareConvolution = SpatialShareConvolution(Int, Int, Int, Int, Int, Int, Int, Int, Int, Boolean, InitializationMethod)
    creating: createSpatialShareConvolution
    '''

    def __init__(self,
                n_input_plane,
                n_output_plane,
                kernel_w,
                kernel_h,
                stride_w=1,
                stride_h=1,
                pad_w=0,
                pad_h=0,
                n_group=1,
                propagate_back=True,
                init_method=Default,
                bigdl_type="float"):
        super(SpatialShareConvolution, self).__init__(None, bigdl_type,
                                                      n_input_plane,
                                                      n_output_plane,
                                                      kernel_w,
                                                      kernel_h,
                                                      stride_w,
                                                      stride_h,
                                                      pad_w,
                                                      pad_h,
                                                      n_group,
                                                      propagate_back,
                                                      init_method)

class SpatialSubtractiveNormalization(Model):
    '''
    >>> spatialSubtractiveNormalization = SpatialSubtractiveNormalization(Int, Tensor)
    creating: createSpatialSubtractiveNormalization
    '''

    def __init__(self,
                n_input_plane,
                kernel=1,
                bigdl_type="float"):
        super(SpatialSubtractiveNormalization, self).__init__(None, bigdl_type,
                                                              n_input_plane,
                                                              kernel)

class SpatialZeroPadding(Model):
    '''
    >>> spatialZeroPadding = SpatialZeroPadding(Int, Int, Int, Int)
    creating: createSpatialZeroPadding
    '''

    def __init__(self,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                bigdl_type="float"):
        super(SpatialZeroPadding, self).__init__(None, bigdl_type,
                                                 pad_left,
                                                 pad_right,
                                                 pad_top,
                                                 pad_bottom)

class Sqrt(Model):
    '''
    >>> sqrt = Sqrt()
    creating: createSqrt
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Sqrt, self).__init__(None, bigdl_type)

class Square(Model):
    '''
    >>> square = Square()
    creating: createSquare
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Square, self).__init__(None, bigdl_type)

class Squeeze(Model):
    '''
    >>> squeeze = Squeeze(Int)
    creating: createSqueeze
    '''

    def __init__(self,
                num_input_dims=Int.MinValue,
                bigdl_type="float"):
        super(Squeeze, self).__init__(None, bigdl_type,
                                      num_input_dims)

class Sum(Model):
    '''
    >>> sum = Sum(Int, Int, Boolean)
    creating: createSum
    '''

    def __init__(self,
                dimension=1,
                n_input_dims=-1,
                size_average=False,
                bigdl_type="float"):
        super(Sum, self).__init__(None, bigdl_type,
                                  dimension,
                                  n_input_dims,
                                  size_average)

class Tanh(Model):
    '''
    >>> tanh = Tanh()
    creating: createTanh
    '''

    def __init__(self,
                bigdl_type="float"):
        super(Tanh, self).__init__(None, bigdl_type)

class TanhShrink(Model):
    '''
    >>> tanhShrink = TanhShrink()
    creating: createTanhShrink
    '''

    def __init__(self,
                bigdl_type="float"):
        super(TanhShrink, self).__init__(None, bigdl_type)

class Threshold(Model):
    '''
    >>> threshold = Threshold(Double, Double, Boolean)
    creating: createThreshold
    '''

    def __init__(self,
                th=1e-6,
                v=0.0,
                ip=False,
                bigdl_type="float"):
        super(Threshold, self).__init__(None, bigdl_type,
                                        th,
                                        v,
                                        ip)

class TimeDistributed(Model):
    '''
    >>> timeDistributed = TimeDistributed()
    creating: createTimeDistributed
    '''

    def __init__(self,
                bigdl_type="float"):
        super(TimeDistributed, self).__init__(None, bigdl_type)

class TimeDistributedCriterion(Model):
    '''
    >>> timeDistributedCriterion = TimeDistributedCriterion(TensorCriterion)
    creating: createTimeDistributedCriterion
    '''

    def __init__(self,
                critrn,
                bigdl_type="float"):
        super(TimeDistributedCriterion, self).__init__(None, bigdl_type,
                                                       critrn)

class Transpose(Model):
    '''
    >>> transpose = Transpose(Array)
    creating: createTranspose
    '''

    def __init__(self,
                permutations,
                bigdl_type="float"):
        super(Transpose, self).__init__(None, bigdl_type,
                                        permutations)

class Unsqueeze(Model):
    '''
    >>> unsqueeze = Unsqueeze(Int, Int)
    creating: createUnsqueeze
    '''

    def __init__(self,
                pos,
                num_input_dims=Int.MinValue,
                bigdl_type="float"):
        super(Unsqueeze, self).__init__(None, bigdl_type,
                                        pos,
                                        num_input_dims)

class Utils(Model):
    '''
    >>> utils = Utils(TensorNumeric)
    creating: createUtils
    '''

    def __init__(self,
                ev,
                bigdl_type="float"):
        super(Utils, self).__init__(None, bigdl_type,
                                    ev)

class View(Model):
    '''
    >>> view = View(Int)
    creating: createView
    '''

    def __init__(self,
                sizes,
                bigdl_type="float"):
        super(View, self).__init__(None, bigdl_type,
                                   sizes)

