require 'nn'

local GPUContainer, parent = torch.class('nn.GPUContainer', 'nn.Container')


function GPUContainer:__init(module)
   parent.__init(self)
   self.modules = {module}
end

function GPUContainer:add()
   local new_module = self.modules[1]:sharedClone()
   if #self.modules == 0 then
      self.gradInput = new_module.gradInput
   end
   table.insert(self.modules, new_module)
   self.output = new_module.output
   return self
end

function GPUContainer:updateOutput(input)
   local currentOutput = input
   for i=1,input:size(3) do
      if self.modules[i] == nil then
         self:add()
      end
      currentOutput = self:rethrowErrors(self.modules[i], i, 'updateOutput', currentOutput)
   end
   self.output = currentOutput
   return currentOutput
end

function GPUContainer:updateGradInput(input, gradOutput)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[input:size(3)]
   for i=input:size(3)-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = self:rethrowErrors(currentModule, i+1, 'updateGradInput', previousModule.output, currentGradOutput)
      currentModule = previousModule
   end
   currentGradOutput = self:rethrowErrors(currentModule, 1, 'updateGradInput', input, currentGradOutput)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function GPUContainer:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local currentGradOutput = gradOutput
   local currentModule = self.modules[input:size(3)]
   for i=input:size(3)-1,1,-1 do
      local previousModule = self.modules[i]
      self:rethrowErrors(currentModule, i+1, 'accGradParameters', previousModule.output, currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end

   self:rethrowErrors(currentModule, 1, 'accGradParameters', input, currentGradOutput, scale)
end

function GPUContainer:backward(input, gradOutput, scale)
   scale = scale or 1
   local currentGradOutput = gradOutput
   local currentModule = self.modules[input:size(3)]
   for i=input:size(3)-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = self:rethrowErrors(currentModule, i+1, 'backward', previousModule.output, currentGradOutput, scale)
      currentModule.gradInput = currentGradOutput
      currentModule = previousModule
   end
   currentGradOutput = self:rethrowErrors(currentModule, 1, 'backward', input, currentGradOutput, scale)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function GPUContainer:accUpdateGradParameters(input, gradOutput, lr)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[input:size(3)]
   for i=input:size(3)-1,1,-1 do
      local previousModule = self.modules[i]
      self:rethrowErrors(currentModule, i+1, 'accUpdateGradParameters', previousModule.output, currentGradOutput, lr)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end

   self:rethrowErrors(currentModule, 1, 'accUpdateGradParameters', input, currentGradOutput, lr)
end


function GPUContainer:__tostring__()
   return 'nn.GPUContainer'
end
