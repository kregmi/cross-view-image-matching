--
-- code derived from https://github.com/kregmi/cross-view-image-synthesis
--

require 'image'
require 'nn'
require 'nngraph'
util = paths.dofile('util/util.lua')
torch.setdefaulttensortype('torch.FloatTensor')
require 'csvigo'


opt = {
   DATA_ROOT = '/home/kregmi/Documents/cvusa_cvpr2017/crossnet-master',         -- path to images (should have subfolders 'train', 'val', etc)
    batchSize = 2,            -- # images in batch
    flip=0,                   -- horizontal mirroring data augmentation
    display = 0,              -- display samples while training. 0 = false
    display_id = 200,         -- display window id.
    gpu = 1,                  -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    how_many = 'all',         -- how many test images to run (set to all to run on every image found in the data/phase folder)
    which_direction = 'g2a', -- g2a or a2g
    phase = 'test',            -- train, val, test ,etc
    preprocess = 'regular',   -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
    aspect_ratio = 1.0,       -- aspect ratio of result images
    name = 'cvusa_synthesis_xfork_g2a',                -- name of experiment, selects which model to run, should generally should be passed on command line
    input_nc = 3,             -- #  of input image channels for img
    output_nc_seg = 3,             -- #  of input image channels for seg
    output_nc = 3,            -- #  of output image channels
    serial_batches = 1,       -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1,    -- iter into serial image list
    cudnn = 1,                -- set to 0 to not use cudnn (untested)
    checkpoints_dir = './checkpoints', -- loads models from here
    results_dir='./results/',          -- saves results here
    which_epoch = '30',            -- which epoch to test? set to 'latest' to use latest cached model
    train_data_path = '/home/kregmi/Documents/cvusa_cvpr2017/crossnet-master/splits/train-19zl.csv',
}


-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.nThreads = 1 -- test only works with 1 thread...
print(opt)

if opt.display == 0 then opt.display = false end

opt.manualSeed = 10 -- set seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

opt.netG_name = opt.name .. '/' .. opt.which_epoch .. '_net_G'


-- translation direction
local input_nc = opt.input_nc
local output_nc_seg = opt.output_nc_seg
local output_nc = opt.output_nc


----------------------------------------------------------------------------
local output_img = torch.Tensor(opt.batchSize, input_nc, 512, 512)     -- real image view 1
local input = torch.Tensor(opt.batchSize, output_nc, 256, 1024)    -- real image view 2
local fake_img = torch.Tensor(opt.batchSize, output_nc + output_nc_seg, 512, 512)



opt.condition_GAN = 0

print('checkpoints_dir', opt.checkpoints_dir)
local netG = util.load(paths.concat(opt.checkpoints_dir, opt.netG_name .. '.t7'), opt)
netG:evaluate()
print(netG)


function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end


batchSize = opt.batchSize

m = csvigo.load({path = opt.train_data_path, mode = "large"})
local number_of_files = #m
if batchSize > number_of_files then batchSize = number_of_files end
num_files = math.floor(number_of_files/batchSize)
number_of_files = num_files * batchSize
print(number_of_files)
print(num_files)
print(batchSize)


--------------------------------------------------------------------------------------------------------------
function load_and_preprocess_regular_image(path)
  local imA = image.load(path, 3, 'float')
  local perm = torch.LongTensor{3, 2, 1}
  imA = imA:index(1, perm)--:mul(256.0): brg, rgb
  imA = imA:mul(2):add(-1)
  assert(imA:max()<=1,"A: badly scaled inputs")
  assert(imA:min()>=-1,"A: badly scaled inputs")
  return imA
end

function deprocess(img)
    -- BGR to RGB
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm)
    -- [-1,1] to [0,1]
    img = img:add(1):div(2)
    return img
end

function deprocess_batch(batch)
  for i = 1, batch:size(1) do
   batch[i] = deprocess(batch[i]:squeeze())
  end
return batch
end
--------------------------------------------------------------------------------------------------------------
function createImageBatch(n)
  filename = {}
  for i = 1, batchSize do
    table.insert(filename, m[n+i-1][1])
    local ground_image = load_and_preprocess_regular_image(opt.DATA_ROOT .. '/' ..  m[n+i-1][2])
    ground_image = image.scale(ground_image, 1024, 256)
    input[i] = ground_image
  end
  return input, filename
end
--------------------------------------------------------------------------------------------------------------

for n=1, number_of_files, batchSize do
    print('processing batch ' .. n)
    
    input, filename = createImageBatch(n)
    filepaths_curr = util.basename_batch(filename)
    print('filepaths_curr: ', filepaths_curr)
    
    
    if opt.gpu > 0 then
        input = input:cuda()
    end
    
    fake_img = netG:forward(input)
    output_img = fake_img[1]

    output_img = deprocess_batch(output_img)
    -- output_seg = deprocess_batch(output[2])

    --input = deprocess_batch(input):float()
    output_img = output_img:float()
    -- output_seg = output_seg:float()
    -- target = util.deprocess_batch(target):float()

    paths.mkdir(paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase))
    local image_dir = paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase, 'images')
    paths.mkdir(image_dir)
    --paths.mkdir(paths.concat(image_dir,'input'))
    paths.mkdir(paths.concat(image_dir,'output'))
    -- paths.mkdir(paths.concat(image_dir,'seg_output'))
    for i=1, opt.batchSize do
        --image.save(paths.concat(image_dir,'input',filepaths_curr[i]), input[i])
        image.save(paths.concat(image_dir,'output',filepaths_curr[i]), output_img[i])
        -- image.save(paths.concat(image_dir,'seg_output',filepaths_curr[i]), output_seg[i])
    end
    print('Saved images to: ', image_dir)
end
