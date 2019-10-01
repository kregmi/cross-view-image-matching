--
-- code derived from https://github.com/soumith/dcgan.torch
-- code derived from https://github.com/kregmi/cross-view-image-synthesis
--

--[[
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).
    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
local matio = require 'matio'
paths.dofile('dataset.lua')
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
-- print(os.getenv('DATA_ROOT'))
--opt.data = paths.concat(os.getenv('DATA_ROOT'), opt.phase)

opt.data = paths.concat(opt.DATA_ROOT_Ah, opt.phase)
opt.data_Sh = paths.concat(opt.DATA_ROOT_Sh, opt.phase)
opt.data_As = paths.concat(opt.DATA_ROOT_As, opt.phase)
opt.data_Ss = paths.concat(opt.DATA_ROOT_Ss, opt.phase)


if not paths.dirp(opt.data) then
    error('Did not find directory: ' .. opt.data)
end
if not paths.dirp(opt.data_Sh) then
    error('Did not find directory: ' .. opt.data_Sh)
end
if not paths.dirp(opt.data_As) then
    error('Did not find directory: ' .. opt.data_As)
end
if not paths.dirp(opt.data_Ss) then
    error('Did not find directory: ' .. opt.data_Ss)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local input_nc = opt.input_nc -- input channels
local output_nc = opt.output_nc
local loadSize   = {input_nc, opt.loadSize}
local sampleSize = {input_nc, opt.fineSize}




function split_filename(s, delimiter)
    result = {};
    for match in (s..delimiter):gmatch("(.-)"..delimiter) do
        table.insert(result, match);
    end
    return result[#result]
end


local function load_two_images_regular(path)
   -- print (path)
   local imA = image.load(path, 3)
   filename = split_filename(path, '/')
   pathB = opt.DATA_ROOT_B .. '/' .. filename
   local imB = image.load(pathB, 3)
   return imA,imB
end

local function load_input_matfiles(path)
  -- body
  -- print (path)
  -- os.exit(1)
    local aerial_hypercolumn = matio.load(opt.DATA_ROOT_Ah .. '/' .. opt.phase .. '/' .. path).hypercolumn
    local street_hypercolumn = matio.load(opt.DATA_ROOT_Sh .. '/' .. opt.phase .. '/' .. path).hypercolumn
    local aerial_segmap = matio.load(opt.DATA_ROOT_As .. '/' .. opt.phase .. '/' .. path).segmap
    local street_segmap = matio.load(opt.DATA_ROOT_Ss .. '/' .. opt.phase .. '/' .. path).segmap

    return aerial_hypercolumn, street_hypercolumn, aerial_segmap, street_segmap

end

local function load_and_preprocess_image_2(path)
   -- print (path)
   local imA = image.load(path, 3)
   -- local rot = torch.random(4) - 1
   imA = image.rotate(imA, (torch.random(4)-1) * 1.5708)
   -- print(path)
   -- os.exit(1)

   local h = imA:size(2)
   local w = imA:size(3)
   filename = split_filename(path, '/')
   pathB = opt.DATA_ROOT_B .. '/' .. filename
-- print (path) 
   local imB = image.load(pathB, 3)
   local imagePixelData_A = torch.DoubleTensor(3,opt.fineSize,opt.fineSize)
   local imagePixelData_B = torch.DoubleTensor(3,opt.fineSize,opt.fineSize)

   imA = image.rgb2yuv(imA):type('torch.DoubleTensor')
   imB = image.rgb2yuv(imB):type('torch.DoubleTensor')

--print (imA:size())
-- print (imagePixelData_A:size())
--os.exit(1)
   for c = 1,3 do
     local v1 = torch.sqrt(torch.var(imA[c]))
     local m1 = torch.mean(imA[c])

     local v2 = torch.sqrt(torch.var(imB[c]))
     local m2 = torch.mean(imB[c])   

     imA[c] = imA[c] - m1
     imA[c] = imA[c] / torch.sqrt(v1)
--     print (imagePixelData_A[c]:size())
--     print (imA[c]:size())
--     print (path)
     imagePixelData_A[{{c}, {}, {}}] = imA[c]

     imB[c] = imB[c] - m2
     imB[c] = imB[c] / torch.sqrt(v2)
     imagePixelData_B[{{c}, {}, {}}] = imB[c]

   end 
-- os.exit(1)
   imAB = torch.cat(imagePixelData_A, imagePixelData_B, 1)

   return imAB
end





local function load_and_preprocess_image(path)
   local input = image.load(path, 3)
   local h = input:size(2)
   local w = input:size(3)
   local imagePixelData_A = torch.DoubleTensor(3,opt.fineSize,opt.fineSize)
   local imagePixelData_B = torch.DoubleTensor(3,opt.fineSize,opt.fineSize)

   local imA = image.crop(input, 0, 0, w/2, h)
   local imB = image.crop(input, w/2, 0, w, h)

   imA = image.rgb2yuv(imA):type('torch.DoubleTensor')
   imB = image.rgb2yuv(imB):type('torch.DoubleTensor')

   for c = 1,3 do
     local v1 = torch.sqrt(torch.var(imA[c]))
     local m1 = torch.mean(imA[c])

     local v2 = torch.sqrt(torch.var(imB[c]))
     local m2 = torch.mean(imB[c])   

     imA[c] = imA[c] - m1
     imA[c] = imA[c] / torch.sqrt(v1)
     imagePixelData_A[{{c}, {}, {}}] = imA[c]

     imB[c] = imB[c] - m2
     imB[c] = imB[c] / torch.sqrt(v2)
     imagePixelData_B[{{c}, {}, {}}] = imB[c]

   end 
   imAB = torch.cat(imagePixelData_A, imagePixelData_B, 1)

   -- return imagePixelData_A, imagePixelData_B
   return imAB
end

-- local load_and_preprocess_image = function(img)

--   local imagePixelData = torch.DoubleTensor(3,opt.fineSize,opt.fineSize)
--   img = image.rgb2yuv(img):type('torch.DoubleTensor')
--   for c = 1,3 do
--      local v = torch.sqrt(torch.var(img[c]))
--      local m = torch.mean(img[c])
--      img[c] = img[c] - m
--      img[c] = img[c] / torch.sqrt(v)
--      imagePixelData[{{c}, {}, {}}] = img[c]
--   end  
--   return imagePixelData
-- end


local preprocessAandB = function(imA, imB)
  imA = image.scale(imA, loadSize[2], loadSize[2])
  imB = image.scale(imB, loadSize[2], loadSize[2])
  local perm = torch.LongTensor{3, 2, 1}
  imA = imA:index(1, perm)--:mul(256.0): brg, rgb
  imA = imA:mul(2):add(-1)
  imB = imB:index(1, perm)
  imB = imB:mul(2):add(-1)
--   print(img:size())
  assert(imA:max()<=1,"A: badly scaled inputs")
  assert(imA:min()>=-1,"A: badly scaled inputs")
  assert(imB:max()<=1,"B: badly scaled inputs")
  assert(imB:min()>=-1,"B: badly scaled inputs")
 
  
  local oW = sampleSize[2]
  local oH = sampleSize[2]
  local iH = imA:size(2)
  local iW = imA:size(3)
  
  if iH~=oH then     
    h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  end
  
  if iW~=oW then
    w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  end
  if iH ~= oH or iW ~= oW then 
    imA = image.crop(imA, w1, h1, w1 + oW, h1 + oH)
    imB = image.crop(imB, w1, h1, w1 + oW, h1 + oH)
  end
  
  if opt.flip == 1 and torch.uniform() > 0.5 then 
    imA = image.hflip(imA)
    imB = image.hflip(imB)
  end
  
  return imA, imB
end



local function loadImageChannel(path)
    local input = image.load(path, 3, 'float')
    input = image.scale(input, loadSize[2], loadSize[2])

    local oW = sampleSize[2]
    local oH = sampleSize[2]
    local iH = input:size(2)
    local iW = input:size(3)
    
    if iH~=oH then     
      h1 = math.ceil(torch.uniform(1e-2, iH-oH))
    end
    
    if iW~=oW then
      w1 = math.ceil(torch.uniform(1e-2, iW-oW))
    end
    if iH ~= oH or iW ~= oW then 
      input = image.crop(input, w1, h1, w1 + oW, h1 + oH)
    end
    
    
    if opt.flip == 1 and torch.uniform() > 0.5 then 
      input = image.hflip(input)
    end
    
--    print(input:mean(), input:min(), input:max())
    local input_lab = image.rgb2lab(input)
--    print(input_lab:size())
--    os.exit()
    local imA = input_lab[{{1}, {}, {} }]:div(50.0) - 1.0
    local imB = input_lab[{{2,3},{},{}}]:div(110.0)
    local imAB = torch.cat(imA, imB, 1)
    assert(imAB:max()<=1,"A: badly scaled inputs")
    assert(imAB:min()>=-1,"A: badly scaled inputs")
    
    return imAB
end

--local function loadImage

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   local h = input:size(2)
   local w = input:size(3)

   local imA = image.crop(input, 0, 0, w/2, h)
   local imB = image.crop(input, w/2, 0, w, h)
   
   return imA, imB
end

local function loadImageInpaint(path)
  local imB = image.load(path, 3, 'float')
  imB = image.scale(imB, loadSize[2], loadSize[2])
  local perm = torch.LongTensor{3, 2, 1}
  imB = imB:index(1, perm)--:mul(256.0): brg, rgb
  imB = imB:mul(2):add(-1)
  assert(imB:max()<=1,"A: badly scaled inputs")
  assert(imB:min()>=-1,"A: badly scaled inputs")
  local oW = sampleSize[2]
  local oH = sampleSize[2]
  local iH = imB:size(2)
  local iW = imB:size(3)
  if iH~=oH then     
    h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  end
  
  if iW~=oW then
    w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  end
  if iH ~= oH or iW ~= oW then 
    imB = image.crop(imB, w1, h1, w1 + oW, h1 + oH)
  end
  local imA = imB:clone()
  imA[{{},{1 + oH/4, oH/2 + oH/4},{1 + oW/4, oW/2 + oW/4}}] = 1.0
  if opt.flip == 1 and torch.uniform() > 0.5 then 
    imA = image.hflip(imA)
    imB = image.hflip(imB)
  end
  imAB = torch.cat(imA, imB, 1)
  return imAB
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   filename = split_filename(path, '/')
   aerial_hypercolumn, street_hypercolumn, aerial_segmap, street_segmap = load_input_matfiles(filename)
   -- print(aerial_hypercolumn:size())
   -- print(street_hypercolumn:size())
   -- print(aerial_segmap:size())
   -- print(street_segmap:size())
   -- os.exit(1)


   local input_feats = {aerial_hypercolumn, street_hypercolumn, aerial_segmap, street_segmap}
   return input_feats
end

--------------------------------------
-- trainLoader
print('trainCache', trainCache)
--if paths.filep(trainCache) then
--   print('Loading train metadata from cache')
--   trainLoader = torch.load(trainCache)
--   trainLoader.sampleHookTrain = trainHook
--   trainLoader.loadSize = {input_nc, opt.loadSize, opt.loadSize}
--   trainLoader.sampleSize = {input_nc+output_nc, sampleSize[2], sampleSize[2]}
--   trainLoader.serial_batches = opt.serial_batches
--   trainLoader.split = 100
--else
print('Creating train metadata')
--   print(opt.data)
print('serial batch:, ', opt.serial_batches)
trainLoader = dataLoader{
    paths = {opt.data},
    loadSize = {input_nc, loadSize[2], loadSize[2]},
    sampleSize = {input_nc+output_nc, sampleSize[2], sampleSize[2]},
    split = 100,
    serial_batches = opt.serial_batches, 
    verbose = true
 }
--   print('finish')
--torch.save(trainCache, trainLoader)
--print('saved metadata cache at', trainCache)
trainLoader.sampleHookTrain = trainHook
--end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end
