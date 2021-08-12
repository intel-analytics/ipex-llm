--
-- Copyright 2016 The BigDL Authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
--
-- This script can generate some tensor files from torch preprocess, which
-- can be used in bigdl test
--

require 'image'

local function loadImage(path)
    local input = image.load(path, 3, 'float')
    return input
end

local function norm(out, mean, std)
    for i = 1, 3 do -- channels
    if mean then out[{ { i }, {}, {} }]:add(-mean[i]) end
    if std then out[{ { i }, {}, {} }]:div(std[i]) end
    end
    return out
end

local function crop(img, oH, oW)
    local iW = img:size(3)
    local iH = img:size(2)
    local h1 = math.ceil(torch.uniform(1e-2, iH - oH))
    local w1 = math.ceil(torch.uniform(1e-2, iW - oW))
    local out = image.crop(img, w1, h1, w1 + oW, h1 + oH)
    return out
end

local function preprocess(source, target)
    local img = loadImage(source)
    img = crop(img, 224, 224)
    img = image.hflip(img)
    norm(img, { 0.4, 0.5, 0.6 }, { 0.1, 0.2, 0.3 })
    torch.save(target, img)
end

torch.manualSeed(100)
preprocess('n02110063/n02110063_11239.JPEG', 'torch/n02110063_11239.t7')
torch.manualSeed(100)
preprocess('n04370456/n04370456_5753.JPEG', 'torch/n04370456_5753.t7')
torch.manualSeed(100)
preprocess('n15075141/n15075141_38508.JPEG', 'torch/n15075141_38508.t7')
torch.manualSeed(100)
preprocess('n99999999/n03000134_4970.JPEG', 'torch/n03000134_4970.t7')
