function createModel(nGPU)
    local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32

    -- Create tables describing VGG configurations A, B, D, E
    local cfg = {}
    if modelType == 'A' then
        cfg = { 64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' }
    elseif modelType == 'B' then
        cfg = { 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' }
    elseif modelType == 'D' then -- vgg16
        cfg = { 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M' }
    elseif modelType == 'E' then -- vgg19
        cfg = { 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M' }
    else
        error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
    end

    local features = nn.Sequential()
    do
        local iChannels = 3;
        for k, v in ipairs(cfg) do
            if v == 'M' then
                features:add(nn.SpatialMaxPooling(2, 2, 2, 2))
            else
                local oChannels = v;
                local conv3 = nn.SpatialConvolution(iChannels, oChannels, 3, 3, 1, 1, 1, 1);
                features:add(conv3)
                features:add(nn.ReLU(true))
                iChannels = oChannels;
            end
        end
    end

    if nGPU > 0 then
        features:cuda()
        features = makeDataParallel(features, nGPU) -- defined in util.lua
    end

    local classifier = nn.Sequential()
    classifier:add(nn.View(512 * 7 * 7))
    classifier:add(nn.Linear(512 * 7 * 7, 4096))
    classifier:add(nn.Threshold(0, 1e-6))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(4096, 4096))
    classifier:add(nn.Threshold(0, 1e-6))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(4096, nClasses))
    classifier:add(nn.LogSoftMax())
    if nGPU > 0 then
        classifier:cuda()
    end

    local model = nn.Sequential()
    model:add(features):add(classifier)
    model.imageSize = 256
    model.imageCrop = 224

    return model
end