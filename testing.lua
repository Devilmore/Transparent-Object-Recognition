require 'torch'  
require 'image'  
require 'nn'     
require 'nnx'      -- provides a normalization operator
require 'lfs'

model = torch.load('results/model.net')
classes = {'cocktailglass','colaglass','shot','waterglass','wineglass','whitebeerglass', 'meh' } --TODO NEED TO SAVE AND LOAD THAT ONE

model:evaluate()

-- test over test data
print'==> testing'
print (classes)


file = "../images/Small/colaglass1.png"      
testimage = torch.Tensor(1, 3, 32, 32)
image.rgb2yuv(testimage,image.load(file))

-- test sample
result = model:forward(testimage)

print(result)

local key = 1
local max = result[key]
k = 1
for k = 1,  result:size(1) do
    if result[k] > max then
        key = k 
        max = result[k]
    end
end

print(key, max)

print(sys.COLORS.red .. classes[key])



