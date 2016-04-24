require 'nn'
require 'optim'
require 'image'
require 'cunn'
require 'cudnn'
require 'NeuralGPU'
require 'dpnn'
require 'GPUContainer'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Neural GPU')
cmd:text()
cmd:text('Options:')
cmd:option('-batchSize', 32, 'batch size')
cmd:option('-seqLen', 5, 'length of sequences')
cmd:option('-gpuSize', 24, 'embedding size')
cmd:option('-gpuWidth', 4, 'gpu width')
cmd:option('-updatePerEpoch', 1000, 'updates per epoch')
cmd:option('-maxEpochs', 1000, 'max number of epochs')
cmd:text()
opt = cmd:parse(arg)

dofile('generator.lua')

local layers = {NeuralGPU(opt.gpuSize, true),
                NeuralGPU(opt.gpuSize, true)}

local neuralGPUStack = nn.Sequential()
for j=1,#layers do
   neuralGPUStack:add(layers[j]:sharedClone())
end

local model = nn.Sequential()
model:add(nn.LookupTable(4, opt.gpuSize))
model:add(nn.Reshape(opt.batchSize, opt.seqLen*2+1, opt.gpuSize, 1))
model:add(nn.Transpose({2,3}))
model:add(nn.SpatialZeroPadding(0, opt.gpuWidth-1, 0, 0))
model:add(nn.GPUContainer(neuralGPUStack))
model:add(cudnn.SpatialConvolution(opt.gpuSize, 4, 1, 1))
model:add(nn.Transpose({2, 3}))
model:add(nn.Select(4, 1))
model:add(nn.Reshape(opt.batchSize * (opt.seqLen*2+1), 4))
model:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()

model:cuda()
criterion:cuda()

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('Using model:')
print(model)

classes = {'0', '1', '+', 'pad'}
confusion = optim.ConfusionMatrix(classes)

-- log results to files
accLogger = optim.Logger(paths.concat('log', 'accuracy.log'))
errLogger = optim.Logger(paths.concat('log', 'error.log'   ))

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   local trainError = 0

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ', seqLen = ' .. opt.seqLen .. ']')
   for t = 1,opt.updatePerEpoch do
      -- disp progress
      xlua.progress(t, opt.updatePerEpoch)

      -- create mini batch
      local inputs, targets = binary_sum_batch(opt.batchSize, opt.seqLen)

      targets = targets:view(opt.batchSize * (opt.seqLen*2+1))

      inputs = inputs:cuda()
      targets = targets:cuda()

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- estimate f
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         for i=1,targets:size(1) do
            -- update confusion
            confusion:add(outputs[i], targets[i])
         end

         trainError = trainError + f

         -- return f and df/dX
         return f,gradParameters
      end

      config = config or {}
      optim.adam(feval, parameters, config)
   end

   -- train error
   trainError = trainError / opt.updatePerEpoch

   -- time taken
   time = sys.clock() - time
   time = time / opt.updatePerEpoch
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   local trainAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- next epoch
   epoch = epoch + 1

   -- apply curriculum
   if trainAccuracy > 90 then
      for j=1,#layers do
         neuralGPUStack:add(layers[j]:sharedClone():cuda())
      end
      for j=1,#layers do
         neuralGPUStack:add(layers[j]:sharedClone():cuda())
      end

      opt.seqLen = opt.seqLen + 1
      model.modules[2] = nn.Reshape(opt.batchSize, opt.seqLen*2+1, opt.gpuSize, 1):cuda()
      model.modules[9] = nn.Reshape(opt.batchSize * (opt.seqLen*2+1), 4):cuda()
   end

   return trainAccuracy, trainError
end

for i=1,opt.maxEpochs do
   trainAcc, trainErr = train()

   -- update logger
   accLogger:add{['% train accuracy'] = trainAcc, ['% test accuracy'] = testAcc}
   errLogger:add{['% train error']    = trainErr, ['% test error']    = testErr}

   -- plot logger
   accLogger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
   errLogger:style{['% train error']    = '-', ['% test error']    = '-'}
   accLogger:plot()
   errLogger:plot()
end
