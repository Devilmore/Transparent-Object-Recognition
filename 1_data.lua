require 'torch'  
require 'image'  
require 'nn'     
require 'nnx'      -- provides a normalization operator
require 'lfs'

if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

images = '../images/Small'

function dirtree(dir)
  assert(dir and dir ~= "", "directory parameter is missing or empty")
  if string.sub(dir, -1) == "/" then
    dir=string.sub(dir, 1, -2)
  end
  local function yieldtree(dir)
    for entry in lfs.dir(dir) do
      if entry ~= "." and entry ~= ".." then
        entry=dir.."/"..entry
	local attr=lfs.attributes(entry)
	coroutine.yield(entry,attr)
	if attr.mode == "directory" then
	  yieldtree(entry)
	end
      end
    end
  end
  return coroutine.wrap(function() yieldtree(dir) end)
end

local opt = opt or {
   visualize = true,
   size = 'small',
   patches='all'
}

print(sys.COLORS.red ..  '==> loading dataset')

size = 0

for file, attr in dirtree(images) do
	if (attr.mode == "file") then
		size = size+1
	end
end

local imagesAll = torch.Tensor(size,3,32,32)
local labelsAll = torch.Tensor(size)

i = 1
j = 1
 	-- add images and labels to the dataset
for file, attr in dirtree(images) do
	if (attr.mode ~= "file") then
	else
		filename = string.format("%s {%s} %s",string.match(file, "(.-)([^/]-([^%.]+))$")) 	-- split path into path,filename,extension and mark filename
		string.gsub(filename,"{(.-)}",function(a) filename = a end)				-- take only the filename
		filename = string.gsub(filename,".png", "") 						-- remove extension
		--print("IMPORTANT SHIT: " .. filename .. " BLA")
		filename = string.gsub(filename,"[".."1234567890".."]",'') 				-- remove numbers, reducing the filename to the label
		print("Loading file: " .. file .. ", will be tagged as \"" .. filename .. "\".")
		-- classes: GLOBAL var!
		classes = {'cocktailglass','colaglass','shot','waterglass','wineglass','whitebeerglass','testcase' }
		if(string.find(filename , "cocktailglass"))then j = 1 
		print("1")
		end
		if(string.find(filename , "colaglass"))then j = 2 
		print("2")
		end
		if(string.find(filename , "shot"))then j = 3
		print("3")		
		end
		if(string.find(filename , "waterglass"))then j = 4
		print("4")		
		end
		if(string.find(filename , "wineglass"))then j = 5 
		print("5")		
		end
		if(string.find(filename , "whitebeerglass"))then j = 6 
		print("6")		
		end
		--image.rgb2yuv(imagesAll[i],image.load(file)) 
	        --itorch.image(imagesAll[i])
	        imagesAll[i] = image.load(file)
		labelsAll[i] = j --filename
		i = i + 1 		
	end
end

-- shuffle dataset: get shuffled indices in this variable:    Why?
local labelsShuffle = torch.randperm((#labelsAll)[1])

local portionTrain = 0.8 -- 80% is train data, rest is test data
local trsize = torch.floor(labelsShuffle:size(1)*portionTrain)
local tesize = labelsShuffle:size(1) - trsize

-- create train set:
trainData = {
   data = torch.Tensor(trsize, 3, 32, 32),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}
--create test set:
testData = {
      data = torch.Tensor(tesize, 3, 32, 32),
      labels = torch.Tensor(tesize),
      size = function() return tesize end
   }

for i=1,trsize do
   trainData.data[i] = imagesAll[labelsShuffle[i]]:clone()
   trainData.labels[i] = labelsAll[labelsShuffle[i]]
end
for i=trsize+1,tesize+trsize do
   testData.data[i-trsize] = imagesAll[labelsShuffle[i]]:clone()
   testData.labels[i-trsize] = labelsAll[labelsShuffle[i]]
end

print '==> preprocessing data'

trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'

for i=1,trsize do
    image.rgb2yuv(trainData.data[i],trainData.data[i])
--    itorch.image(trainData.data[i])
end
for i=1,tesize do
    image.rgb2yuv(testData.data[i],testData.data[i])
--    itorch.image(trainData.data[i])
end
