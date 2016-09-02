#!/bin/bash

echo -n >TorchPerform.csv
echo -e "ReLUPerform Begin"
th ReLUPerform.lua
echo -e "ReLUPerform finish\nSpatialConvolutionPerform Begin"
th SpatialConvolutionPerform.lua
echo -e "SpatialConvolutionPerform finish\nMaxPoolingPerform Begin"
th SpatialMaxPoolingPerform.lua
echo -e "MaxPoolingPerform finish\nLinearPerform Begin"
th LinearPerform.lua
echo -e "LinearPerform finish\nClassNLLCriterionPerform Begin"
th ClassNLLCriterionPerform.lua
echo -e "ClassNLLCriterionPerform finish\nBatchNormalizationPerform Begin"
th BatchNormalizationPerform.lua
echo -e "BatchNormalizationPerform finish\nDropoutPerform Begin"
th DropoutPerform.lua
echo -e "DropoutPerform finish\nBCECriterionPerform Begin"
th BCECriterionPerform.lua
echo -e "BCECriterionPerform finish\nAll finished, use vim TorchPerform.csv to check the torch performance"

