require 'nn'
require 'nngraph'

function NeuralGPU(size)
   local s = nn.Identity()()
   
   local u = nn.Sigmoid()(
                     nn.SpatialConvolutionMM(size, size, 3, 3, 1, 1, 1, 1)(s))
   
   local one_minus_u = nn.AddConstant(1, false)(nn.MulConstant(-1, false)(u))
   
   local r = nn.Sigmoid()(
                     nn.SpatialConvolutionMM(size, size, 3, 3, 1, 1, 1, 1)(s))

   local r_s = nn.CMulTable()({r, s})

   local new_s = nn.Tanh()(
                     nn.SpatialConvolutionMM(size, size, 3, 3, 1, 1, 1, 1)(r_s))
                     
   local u_s = nn.CMulTable()({u, s})
   
   local one_minus_u_new_s = nn.CMulTable()({one_minus_u, new_s})
   
   local sum_s = nn.CAddTable()({u_s, one_minus_u_new_s})
   
   return nn.gModule({s}, {sum_s})
end

return NeuralGPU
