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
cmd:option('-batchSize', 16, 'batch size')
cmd:option('-maxLen', 20, 'length of sequences')
cmd:option('-gpuSize', 24, 'embedding size')
cmd:option('-gpuWidth', 4, 'gpu width')
cmd:option('-updatePerEpoch', 1000, 'updates per epoch')
cmd:option('-maxEpochs', 1000, 'max number of epochs')
cmd:text()
opt = cmd:parse(arg)

dofile('generator.lua')

local layers = {NeuralGPU(opt.gpuSize, false),
                NeuralGPU(opt.gpuSize, false)}

local neuralGPUStack = nn.Sequential()
for j=1,#layers do
   neuralGPUStack:add(layers[j])
end
--neuralGPUStack:add(nn.Dropout(0.05))
local model = nn.Sequential()
model:add(nn.LookupTable(4, opt.gpuSize))
model:add(nn.Reshape(-1, opt.gpuSize, 1, true))
model:add(nn.Transpose({2,3}))
model:add(nn.SpatialZeroPadding(0, opt.gpuWidth-1, 0, 0))
model:add(nn.GPUContainer(neuralGPUStack))
model:add(nn.Transpose({2, 3}))
model:add(nn.Select(4, 1))
model:add(nn.Reshape(-1, opt.gpuSize, false))
model:add(nn.Linear(opt.gpuSize, 4))
model:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()
criterion.sizeAverage = false

model:cuda()
criterion:cuda()

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

print(#parameters)
-- verbose
print('Using model:')
print(model)

classes = {'0', '1', '+', 'pad'}
confusion = optim.ConfusionMatrix(classes)

-- log results to files
accLogger = optim.Logger(paths.concat('log', 'accuracy.log'))
errLogger = optim.Logger(paths.concat('log', 'error.log'   ))

currMaxLen = 1
trainLen = 1

-- training function
function train(epoch)
   -- local vars
   local time = sys.clock()
   local trainError = 0

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ', seqLen = ' .. currMaxLen .. ']')
   for t = 1,opt.updatePerEpoch do
      -- disp progress
      xlua.progress(t, opt.updatePerEpoch)

      -- create mini batch

      if math.random() < 0.8 then
         trainLen = currMaxLen
      else
         trainLen = math.random(currMaxLen)
      end

      local inputs, targets = binary_sum_batch(opt.batchSize, trainLen)

      targets = targets:view(opt.batchSize * (trainLen*2+1))

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

         gradParameters:clamp(-1, 1)
         -- return f and df/dX
         return f,gradParameters
      end

      config = config or {learningRate=1e-3, epsilon = 1e-3}
      optim.adam(feval, parameters, config)
   end

   -- train error
   trainError = trainError / opt.updatePerEpoch

   -- time taken
   time = sys.clock() - time
   time = time / opt.updatePerEpoch
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
   print("Gradient norm = " .. gradParameters:norm())

   -- print confusion matrix
   print(confusion)
   local trainAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- apply curriculum
   if trainAccuracy > 90 then
      currMaxLen = math.min(currMaxLen + 1, opt.maxLen)
   end

   return trainAccuracy, trainError
end

-- training function
function test(epoch, currMaxLen)
   -- local vars
   local time = sys.clock()
   local testError = 0

   -- do one epoch
   print('<trainer> on test set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ', seqLen = ' .. currMaxLen .. ']')
   for t = 1,opt.updatePerEpoch do
      -- disp progress
      xlua.progress(t, opt.updatePerEpoch)

      -- create mini batch

      local inputs, targets = binary_sum_batch(opt.batchSize, currMaxLen)

      targets = targets:view(opt.batchSize * (currMaxLen*2+1))

      inputs = inputs:cuda()
      targets = targets:cuda()

      -- estimate f
      local outputs = model:forward(inputs)

      for i=1,targets:size(1) do
         -- update confusion
         confusion:add(outputs[i], targets[i])
      end
   end

   -- train error
   testError = testError / opt.updatePerEpoch

   -- time taken
   time = sys.clock() - time
   time = time / opt.updatePerEpoch
   print("<trainer> time to evaluate 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   local testAccuracy = confusion.totalValid * 100
   confusion:zero()

   return testAccuracy, testError
end

testLens = {20, 40, 60, 80}
testAccs = {}
testErrs = {}

for j=1,#testLens do
   testAccs[j] = {}
   testErrs[j] = {}
end

for epoch=1,opt.maxEpochs do
   model:training()
   trainAcc, trainErr = train(epoch)

   if epoch % 10 == 0 then
      accsPlot = {}
      errsPlot = {}

      for j=1,#testLens do
         model:evaluate()
         testAcc, testErr = test(epoch, testLens[j])
         testAccs[j][epoch/10] = testAcc
         testErrs[j][epoch/10] = testErr
         accsPlot[j] = {tostring(testLens[j]), torch.Tensor(testAccs[j]), '-'}
         errsPlot[j] = {tostring(testLens[j]), torch.Tensor(testErrs[j]), '-'}
      end
      require 'gnuplot'
      gnuplot.plot(accsPlot)
   end
end
