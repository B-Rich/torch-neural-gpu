function binary_sum_batch(batchSize, n)
   --[[
   0 - 1
   1 - 2
   + - 3
   Pad - 4
   ]]
   local input = torch.Tensor(batchSize, 2*n+1):fill(3)
   local target = torch.Tensor(batchSize, 2*n+1):fill(3)

   input[{{},{1,n}}]:random(0, 1):round()
   input[{{},n+1}]:fill(2)
   input[{{},{n+2,2*n+1}}]:random(0, 1):round()
   
   local reminder = torch.zeros(batchSize)
   for i=1,n do
      target[{{},i}] = input[{{},i}]+input[{{},i+n+1}]+reminder
      reminder:copy(target[{{},i}] / 2):floor()
      target[{{},i}]:copy(target[{{},i}] % 2)
   end

   input:add(1)
   target:add(1)
   
   return input, target
end
