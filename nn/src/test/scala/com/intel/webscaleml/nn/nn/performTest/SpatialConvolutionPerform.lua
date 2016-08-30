--
-- Created by yansh on 16 - 6 - 15.
--

require 'nn'

forwardIterations = 10
backwardIterations = 10
seed = 100

batchSize = 10


allTestCases = {
    -- AlexNet
    {IP = 3, OP = 64, kW = 11, kH = 11, dW = 4, dH = 4, padW = 2, padH = 2, iW = 224, iH = 224},
    {IP = 64, OP = 192, kW = 5, kH = 5, dW = 1, dH = 1, padW = 2, padH = 2, iW = 25, iH = 25},
    {IP = 191, OP = 384, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 12, iH = 12},
    {IP = 384, OP = 256, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 6, iH = 6},
    {IP = 256, OP = 256, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 3, iH = 3},
    --Cifar
    {IP = 3, OP = 64, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 224, iH = 224},
    {IP = 64, OP = 64, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 110, iH = 110},
    {IP = 64, OP = 128, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 54, iH = 54},
    {IP = 128, OP = 128, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1 ,padH = 1 ,iW = 26, iH = 26},
    {IP = 128, OP = 256, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 13, iH = 13},
    {IP = 256, OP = 256, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 6, iH = 6},
    {IP = 256, OP = 512, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 3, iH = 3},
    {IP = 512, OP = 512, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 2, iH = 2},
    
    --GoogleNet
    {IP = 3, OP = 64, kW = 7, kH = 7, dW = 2, dH = 2, padW = 3, padH = 3, iW = 224, iH = 224},
    {IP = 64, OP = 64, kW = 1, kH = 1, dW = 1, dH = 1, padW = 0, padH = 0, iW = 54, iH = 54},
    {IP = 64, OP = 192, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 27, iH = 27},
    {IP = 192, OP = 576, kW = 3, kH = 3, dW = 1, dH = 1, padW = 1, padH = 1, iW = 12, iH = 12},
    {IP = 576, OP = 576, kW = 2, kH = 2, dW = 2, dH = 2, padW = 0, padH = 0, iW = 4, iH = 4}
}

torch.manualSeed(seed)
file = io.open("TorchPerform.csv", "a")
io.output(file)


for _, testCase in pairs(allTestCases) do

    local input = torch.Tensor(batchSize, testCase.IP, testCase.iW, testCase.iH)

    input:apply(function()
        i = torch.uniform(0, 1)
        return i
    end)

    local model = nn.SpatialConvolution(testCase.IP, testCase.OP,
        testCase.kW, testCase.kH, testCase.dW, testCase.dH, testCase.padW, testCase.padH)
    local timer = torch.Timer()
    for j = 1, forwardIterations do
        model:forward(input)
    end

    local timeSpend = timer:time().real / forwardIterations

    file:write("SpatialConvolution(" .. testCase.IP .. "," .. testCase.OP .. "," .. testCase.kW .."," ..
            testCase.kH .. "," .. testCase.dW .. "," .. testCase.dH .. "," .. testCase.padW .. "," .. testCase.padH .. ")forward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")
end

for _, testCase in pairs(allTestCases) do

    local input = torch.Tensor(batchSize, testCase.IP, testCase.iW, testCase.iH)
    input:apply(function()
        i = torch.uniform(0, 1)
        return i
    end
    )

    local model = nn.SpatialConvolution(testCase.IP, testCase.OP,
        testCase.kW, testCase.kH, testCase.dW, testCase.dH, testCase.padW, testCase.padH)
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
    file:write("SpatialConvolution(" .. testCase.IP .. "," .. testCase.OP .. "," .. testCase.kW .."," ..
            testCase.kH .. "," .. testCase.dW .. "," .. testCase.dH .."," .. testCase.padW .. "," .. testCase.padH .. ")backward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")
end

file:close()
