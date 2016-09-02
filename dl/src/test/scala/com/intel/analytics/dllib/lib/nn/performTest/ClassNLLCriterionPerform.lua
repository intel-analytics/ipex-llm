--
-- Created by yao on 6/3/16
--

--require 'torch'
require 'nn'

forwardIterations = 10
backwardIterations = 10
inputNum = 512
featureDim = 512
seed = 100
torch.manualSeed(seed)
file = io.open("TorchPerform.csv", "a")
io.output(file)

output = torch.Tensor(inputNum, featureDim)
target = torch.Tensor(inputNum)
target:apply(function()
    i = torch.ceil(torch.uniform(1, 10))
    return i
end)

criterion = nn.ClassNLLCriterion()

--warm up
for j = 1, forwardIterations do
    criterion:forward(output, target)
end

timer = torch.Timer()
for j = 1, forwardIterations do
    criterion:forward(output, target)
end
timeSpend = timer:time().real / forwardIterations
file:write("ClassNLLCriterion_forward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")

timer = torch.Timer()
for j = 1, backwardIterations do
    criterion:backward(output, target)
end
timeSpend = timer:time().real / backwardIterations
file:write("ClassNLLCriterion_backward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")

file:close()


