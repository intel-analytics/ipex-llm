--
-- Created by yansh on 16-6-22.
--

require 'nn'

forwardIterations = 10
backwardIterations = 10
forwardTimeoutMillis = 30
backwardTimeoutMillis = 120
inputNum = 100
featureDim = 512
seed = 100

torch.manualSeed(seed)
file = io.open("TorchPerform.csv", "a")
io.output(file)

bn = nn.BatchNormalization(featureDim)

for i = 1, featureDim do
    bn.weight[i] = 0.1 * i
    bn.bias[i] = 0.1 * i
end

input = torch.Tensor(inputNum, featureDim)

--warm up
for j = 1, forwardIterations do
    bn:forward(input)
end

timer = torch.Timer()
for j = 1, forwardIterations do
    bn:forward(input)
end
timeSpend = timer:time().real / forwardIterations
file:write("BatchNormalization_forward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")

grads = torch.Tensor(inputNum, featureDim)

timer = torch.Timer()
for j = 1, backwardIterations do
    bn:backward(input,grads)
end
timeSpend = timer:time().real / backwardIterations
file:write("BatchNormalization_backward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")

file:close()

