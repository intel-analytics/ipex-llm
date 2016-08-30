--
-- Created by: yao on 6/6/16
--

require 'nn'

forwardIterations = 10
backwardIterations = 10
batchSize = 100
nInputPlane = 3
height = 100
weight = 512
seed = 100
allTestCases = {
    {kW = 3, kH = 3, dW = 2, dH = 2}, -- AlexNet, GoogleNet
    {kW = 2, kH = 2, dW = 2, dH = 2}, -- Cifar, CifarLocal
    {kW = 3, kH = 3, dW = 1, dH = 1}, -- GoogleNet
    {kW = 3, kH = 3, dW = 3, dH = 3}, -- MNIST
}
torch.manualSeed(seed)
file = io.open("TorchPerform.csv", "a")
io.output(file)

input = torch.Tensor(batchSize, nInputPlane, height, weight)
input:apply(function()
    i = torch.uniform(0, 1)
    return i
end)

for k, testCase in pairs(allTestCases) do
    local model = nn.SpatialMaxPooling(testCase.kW, testCase.kH, testCase.dW, testCase.dH)
    local timer = torch.Timer()
    for j = 1, forwardIterations do
        model:forward(input)
    end
    local timeSpend = timer:time().real / forwardIterations
    file:write("SpatialMaxPooling(" .. testCase.kW .."," ..
        testCase.kH .. "," .. testCase.dW .. "," .. testCase.dH .. ")forward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")
end

for k, testCase in pairs(allTestCases) do
    local model = nn.SpatialMaxPooling(testCase.kW, testCase.kH, testCase.dW, testCase.dH)
    local output = model:forward(input)
    local grad = output:clone():apply(function()
        i = torch.uniform(-2, 2)
        return i
    end)
    local timer = torch.Timer()
    for j = 1, backwardIterations do
        model:backward(input, grad)
    end
    local timeSpend = timer:time().real / backwardIterations
    file:write("SpatialMaxPooling(" .. testCase.kW .."," ..
            testCase.kH .. "," .. testCase.dW .. "," .. testCase.dH .. ")backward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")
end

file:close()

