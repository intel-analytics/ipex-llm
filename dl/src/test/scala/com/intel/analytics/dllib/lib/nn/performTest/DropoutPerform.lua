--
-- Created by yansh on 16-6-22.
--

require 'nn'

forwardIterations = 10
backwardIterations = 10
inputNum = 1000
featureDim = 512
seed = 100
allTestCases = {
    {p = 0.3, forwardTimeoutMillis = 32, backwardTimeoutMillis = 19},
    {p = 0.4, forwardTimeoutMillis = 35, backwardTimeoutMillis = 20},
    {p = 0.5, forwardTimeoutMillis = 32, backwardTimeoutMillis = 20}
}

torch.manualSeed(seed)
file = io.open("TorchPerform.csv", "a")
io.output(file)

input = torch.Tensor(inputNum, featureDim)

for _, testCase in pairs(allTestCases) do
    local model = nn.Dropout(testCase.p)
    local timer = torch.Timer()
    for j = 1, forwardIterations do
        model:forward(input)
    end
    local timeSpend = timer:time().real / forwardIterations
    file:write("Dropout(" .. testCase.p ..  ")forward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")
end

for _, testCase in pairs(allTestCases) do
    local model = nn.Dropout(testCase.p)
    local output = model:forward(input)
    local grads = output:clone()
    local timer = torch.Timer()
    for j = 1, backwardIterations do
        model:backward(input, grads)
    end
    local timeSpend = timer:time().real / backwardIterations
    file:write("Dropout(" .. testCase.p .. ")backward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")
end

file:close()
