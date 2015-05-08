require 'torch'  
require 'image'  
require 'nn'     
require 'nnx'      -- provides a normalization operator
require 'lfs'
require 'json'

cmd = torch.CmdLine()
file = "../images/Small/artificial light - 120cm - 20 degrees - 70cm/cocktailglass1_phatch.png"      

cmd:option('-save', file, 'subdirectory to save/log experiments in')
cmd:text()
opt = cmd:parse(arg or {})
file = opt.save

model = torch.load('results/model.net')
local labelsFile = '../images/tensors/labels.txt'
f = io.open(labelsFile) 
classes = json.decode(f:read("*all")) 

model:evaluate()

-- test over test data
--print'==> testing'
--print (classes)



testimage = torch.Tensor(1, 3, 32, 32)
image.rgb2yuv(testimage,image.load(file))

-- test sample
result = model:forward(testimage)

--print(result)

local key = 1
local max = result[key]
k = 1
for k = 1,  result:size(1) do
    if result[k] > max then
        key = k 
        max = result[k]
    end
end

--print(key, max)

print(classes[key])



