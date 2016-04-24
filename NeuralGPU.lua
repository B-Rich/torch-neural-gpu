require 'nn'
require 'nngraph'
require 'dpnn'

function NeuralGPU(size, bn)
   local s = nn.Identity()()

   local bn_u = nn.Identity()
   local bn_r = nn.Identity()
   local bn_s = nn.Identity()

   if bn == true then
      bn_u = nn.SpatialBatchNormalization(size)
      bn_u.dpnn_parameters = {'weight', 'bias'}

      bn_r = nn.SpatialBatchNormalization(size)
      bn_r.dpnn_parameters = {'weight', 'bias'}

      bn_s = nn.SpatialBatchNormalization(size)
      bn_s.dpnn_parameters = {'weight', 'bias'}
   end

   local u = nn.Sigmoid()(
                     bn_u(cudnn.SpatialConvolution(size, size, 3, 3, 1, 1, 1, 1)(s)))

   local one_minus_u = nn.AddConstant(1, false)(nn.MulConstant(-1, false)(u))

   local r = nn.Sigmoid()(
                     bn_r(cudnn.SpatialConvolution(size, size, 3, 3, 1, 1, 1, 1)(s)))

   local r_s = nn.CMulTable()({r, s})

   local new_s = nn.Tanh()(
                     bn_s(cudnn.SpatialConvolution(size, size, 3, 3, 1, 1, 1, 1)(r_s)))

   local u_s = nn.CMulTable()({u, s})

   local one_minus_u_new_s = nn.CMulTable()({one_minus_u, new_s})

   local sum_s = nn.CAddTable()({u_s, one_minus_u_new_s})

   return nn.gModule({s}, {sum_s})
end

return NeuralGPU
