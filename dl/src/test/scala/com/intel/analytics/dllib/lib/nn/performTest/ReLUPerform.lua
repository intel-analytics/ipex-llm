--
-- Created by yansh on 16-6-22.
--

require "nn"

forwardIterations = 10
backwardIterations = 10
forwardTimeoutMillisIP = 60
backwardTimeoutMillisIP = 65
forwardTimeoutMillisNIP = 170
backwardTimeoutMillisNIP = 400
inputNum = 100
featureNum = 512
seed = 100

torch.manualSeed(seed)
file = io.open("TorchPerform.csv", "a")
io.output(file)

input = torch.Tensor(inputNum, featureNum, 512)

--AlexNet, Cifar, GoogleNet
ipmodel = nn.ReLU(true)
--GoogleNet
nipmodel = nn.ReLU()

--Warm up
for j = 1, forwardIterations do
    ipmodel:forward(input)
end

--ReLU(inplace)

timer = torch.Timer()
for j = 1, forwardIterations do
    ipmodel:forward(input)
end
timeSpend = timer:time().real / forwardIterations
file:write("ReLU(ip)_forward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")

grads = torch.Tensor(inputNum, featureNum, 512)

timer = torch.Timer()
for j = 1, backwardIterations do
    ipmodel:backward(input, grads)
end
timeSpend = timer:time().real / backwardIterations
file:write("ReLU(ip)_backward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")

--ReLU(not inplace)

timer = torch.Timer()
for j = 1, forwardIterations do
    nipmodel:forward(input)
end
timeSpend = timer:time().real / forwardIterations
file:write("ReLU(nip)_forward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")

timer = torch.Timer()
for j = 1, backwardIterations do
    nipmodel:backward(input, grads)
end
timeSpend = timer:time().real / backwardIterations
file:write("ReLU(nip)_backward:" .. string.format("%.6f", timeSpend) * 1000 .. "\n")

file:close()

