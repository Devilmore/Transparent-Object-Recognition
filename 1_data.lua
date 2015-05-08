require 'torch'  
require 'image'  
require 'nn'     
require 'nnx'      -- provides a normalization operator
require 'lfs'
require 'json'

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

-- Define functions we need because lua sucks

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

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

-- define classes glbally
classes = {}
classes_tmp = {}

print(sys.COLORS.red ..  '==> loading dataset')

local trainFile = '../images/tensors/train.t7'
local testFile = '../images/tensors/test.t7'
local labelsFile = '../images/tensors/labels.txt'

-- Load tensors if they exist
if (file_exists(trainFile) and file_exists(testFile) and file_exists(labelsFile)) then
	-- copy train tensor
	loaded = torch.load(trainFile,'binary')
	trsize = loaded.data:size()[1]
	trainData = {
		data = loaded.data,
		labels = loaded.labels,
		size = function() return trsize end
		}
	-- copy test tensor
	loaded = torch.load(testFile,'binary')
	tesize = loaded.data:size()[1]
	testData = {
		data = loaded.data,
		labels = loaded.labels,
		size = function() return tesize end
		}
	--- load label data from labels file
	f = io.open(labelsFile) 
	classes = json.decode(f:read("*all")) 
	f:close() 
-- or create new ones if none (or only one) exist
else
	-- path to images
	images = '../images/Small'
	
	-- determine the tensor size from the number of images and add all filenames to the classes list
	size = 0
	for file, attr in dirtree(images) do
		if (attr.mode == "file") then
			size = size+1
			filename = string.format("%s {%s} %s",string.match(file, "(.-)([^/]-([^%.]+))$"))
			string.gsub(filename,"{(.-)}",function(a) filename = a end)				
			filename = string.gsub(filename,".png", "")
			filename = string.gsub(filename,"[".."1234567890".."]",'')
			table.insert(classes_tmp, filename)
		end
	end

	-- delete duplicate entries in classes list
	local hash = {}
	for _,v in ipairs(classes_tmp) do
		if (not hash[v]) then
			classes[#classes+1] = v
		hash[v] = true
		end
	end

	-- set up variables for image reading loop
	local imagesAll = torch.Tensor(size,3,32,32)
	local labelsAll = torch.Tensor(size)

	local i = 1
	local j = 1

 	-- add images and labels to the dataset
	for file, attr in dirtree(images) do
		if (attr.mode ~= "file") then
		else
			filename = string.format("%s {%s} %s",string.match(file, "(.-)([^/]-([^%.]+))$")) 	-- split path into path,filename,extension and mark filename
			string.gsub(filename,"{(.-)}",function(a) filename = a end)				-- take only the filename
			filename = string.gsub(filename,".png", "") 						-- remove extension
			filename = string.gsub(filename,"[".."1234567890".."]",'') 				-- remove numbers, reducing the filename to the label
			print("Loading file: " .. file .. ", will be tagged as \"" .. filename .. "\".")
			for i=1,table.getn(classes) do
				if (string.find(filename, classes[i])) then
					j = i
				end
			end
			imagesAll[i] = image.load(file)
			labelsAll[i] = j --filename
			i = i + 1 		
		end
	end

	-- shuffle dataset: get shuffled indices in this variable:    Why?
	local labelsShuffle = torch.randperm((#labelsAll)[1])

	local portionTrain = 0.8 -- 80% is train data, rest is test data
	trsize = torch.floor(labelsShuffle:size(1)*portionTrain)
	tesize = labelsShuffle:size(1) - trsize

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

	-- save the created tensors. Classes are written to a labels file for easier use
	print("==> saving tensors in train.t7 with size " .. trsize .. " and test.t7 with size " .. tesize)
	torch.save("../images/tensors/train.t7", trainData, 'binary')
	torch.save("../images/tensors/test.t7", testData, 'binary')
	f = io.open("../images/tensors/labels.txt", "w") 
	f:write(json.encode(classes)) 
	f:close() 
end

print '==> preprocessing data'

trainData.data = trainData.data:double()
testData.data = testData.data:double()

-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'

for i=1,trsize do
    image.rgb2yuv(trainData.data[i],trainData.data[i])
end
for i=1,tesize do
    image.rgb2yuv(testData.data[i],testData.data[i])
end

-- Additional pre-processing which may or may not be useful and visualization which probably wont work.
-- Use itorch instead.

---- Name channels for convenience
--channels = {'y','u','v'}

---- Normalize each channel, and store mean/std
---- per channel. These values are important, as they are part of
---- the trainable parameters. At test time, test data will be normalized
---- using these values.
--print '==> preprocessing data: normalize each feature (channel) globally'
--mean = {}
--std = {}
--for i,channel in ipairs(channels) do
--   -- normalize each channel globally:
--   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
--   std[i] = trainData.data[{ {},i,{},{} }]:std()
--   trainData.data[{ {},i,{},{} }]:add(-mean[i])
--   trainData.data[{ {},i,{},{} }]:div(std[i])
--end

---- Normalize test data, using the training means/stds
--for i,channel in ipairs(channels) do
--   -- normalize each channel globally:
--   testData.data[{ {},i,{},{} }]:add(-mean[i])
--   testData.data[{ {},i,{},{} }]:div(std[i])
--end

---- Local normalization
--print '==> preprocessing data: normalize all three channels locally'

---- Define the normalization neighborhood:
--neighborhood = image.gaussian1D(13)

---- Define our local normalization operator (It is an actual nn module, 
---- which could be inserted into a trainable model):
--normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

---- Normalize all channels locally:
--for c in ipairs(channels) do
--   for i = 1,trainData:size() do
--      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
--   end
--   for i = 1,testData:size() do
--      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
--   end
--end

------------------------------------------------------------------------
--print '==> verify statistics'

---- It's always good practice to verify that data is properly
---- normalized.

--for i,channel in ipairs(channels) do
--   trainMean = trainData.data[{ {},i }]:mean()
--   trainStd = trainData.data[{ {},i }]:std()

--   testMean = testData.data[{ {},i }]:mean()
--   testStd = testData.data[{ {},i }]:std()

--   print('training data, '..channel..'-channel, mean: ' .. trainMean)
--   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

--   print('test data, '..channel..'-channel, mean: ' .. testMean)
--   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
--end

------------------------------------------------------------------------
----print '==> visualizing data'

---- Visualization is quite easy, using itorch.image().

----if opt.visualize then
----   if itorch then
----   first256Samples_y = trainData.data[{ {1,256},1 }]
----   first256Samples_u = trainData.data[{ {1,256},2 }]
----   first256Samples_v = trainData.data[{ {1,256},3 }]
----   itorch.image(first256Samples_y)
----   itorch.image(first256Samples_u)
----   itorch.image(first256Samples_v)
----   else
----      print("For visualization, run this script in an itorch notebook")
----   end
----end
