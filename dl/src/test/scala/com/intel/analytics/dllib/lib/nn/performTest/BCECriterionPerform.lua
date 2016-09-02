--
-- Created by yansh on 16 - 6 - 15.
--

require 'nn'

forwardIterations = 10
backwardIterations = 10
forwardTimeoutMillis = 13
backwardTimeoutMillis = 10
inputNum = 100
featureDim = 512
seed = 100

torch.manualSeed(seed)
file = io.open("TorchPerform.csv", "a")
io.output(file)

criterion = nn.BCECriterion()
input = torch.Tensor(inputNum, featureDim)
target = torch.Tensor(inputNum, featureDim)

timer = torch.Timer()
for j = 1, forwardIterations do
    criterion:forward(input, target)
end
timeSpend = timer:time().real / forwardIterations
file:write("BCECriterion_forward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")

timer = torch.Timer()
for j = 1, backwardIterations do
    criterion:backward(input, target)
end
timeSpend = timer:time().real / backwardIterations
file:write("BCECriterion_backward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")

file:close()

