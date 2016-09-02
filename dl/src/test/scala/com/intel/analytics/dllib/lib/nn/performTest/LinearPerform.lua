--
-- Created by yansh on 16-6-22.
--

require "nn"

forwardIterations = 10
backwardIterations = 10
inputNum = 100
seed = 100
allTestCases = {
    {inputSize = 256 * 6 * 6, outputSize = 4096},--, 3000, 3900},
    {inputSize = 4096, outputSize = 4096},--1500, 2000},
    {inputSize = 256 * 5 * 5, outputSize = 128},-- 65, 90},
    {inputSize = 512, outputSize = 512},-- 21, 30},
    {inputSize = 512, outputSize = 10},-- 2, 2},
    {inputSize = 28 * 4 * 4, outputSize = 768}-- 28, 40}
}

torch.manualSeed(seed)
file = io.open("TorchPerform.csv", "a")
io.output(file)

for _, testCase in pairs(allTestCases) do
    local input = torch.Tensor(inputNum, testCase.inputSize)
    local model = nn.Linear(testCase.inputSize, testCase.outputSize)

    local timer = torch.Timer()
    for j = 1, forwardIterations do
        model:forward(input)
    end
    local timeSpend = timer:time().real / forwardIterations
    file:write("Linear(" .. testCase.inputSize .."," ..
            testCase.outputSize  .. ")forward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")
end

for _, testCase in pairs(allTestCases) do
    local model = nn.Linear(testCase.inputSize,testCase.outputSize)
    local input = torch.Tensor(inputNum, testCase.inputSize)
    local output = model:forward(input)
    local grads = output:clone()

    local timer = torch.Timer()
    for j = 1, backwardIterations do
        model:backward(input, grads)
    end
    local timeSpend = timer:time().real / backwardIterations

    file:write("Linear(" .. testCase.inputSize .."," ..
            testCase.outputSize  .. ")backward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")
end

file:close()
