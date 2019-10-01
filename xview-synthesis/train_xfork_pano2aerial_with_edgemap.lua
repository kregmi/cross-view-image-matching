--
-- code derived from https://github.com/kregmi/cross-view-image-synthesis
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'
require 'cudnn'
require 'cunn'
require 'csvigo'



local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'

opt = {
   DATA_ROOT = '/home/kregmi/Documents/cvusa_cvpr2017/crossnet-master',         -- path to images (should have subfolders 'train', 'val', etc)
   batchSize = 16,          -- # images in batch
   -- loadSize = 256,         -- scale images to this size
   -- fineSize = 256,         --  then crop to this size
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   input_nc = 3,           -- #  of input image channels
   output_nc = 3,          -- #  of output image channels
   output_nc_seg = 3,          -- #  of output image channels
   niter = 100,            -- #  of iter at starting learning rate  -- was 200
   lr = 0.0002,            -- initial learning rate for adam
   beta = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 1,               -- if flip the images for data argumentation
   display = 0,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X, gpu = 1 used generally
   name = 'cvusa_synth_with_edgemap',              -- name of the experiment, should generally be passed on the command line
   phase = 'aerial',             -- train, val, test, etc
   nThreads = 4,                -- # threads for loading data
   save_epoch_freq = 1,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 5000,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 50,             -- print the debug information every print_freq iterations
   display_freq = 100,          -- display the current results every display_freq iterations
   save_display_freq = 5000,    -- save the current display of results every save_display_freq_iterations
   continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   serial_batch_iter = 1,       -- iter into serial image list
   checkpoints_dir = './checkpoints', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   which_model_netG = 'forkG_cvusa',  -- selects model to use for netG  crossview_G_64
   which_model_netD = 'basic_D',  -- selects model to use for netG  crossview_G_64
   which_epoch = '0',            -- epoch number to resume training, used only if continue_train=1
   finetune=1,    -- 1 to load weights from encoders and start training, 0 to resume training 
   lambda1=100,
   lambda2=1,
   train_data_path = '/home/kregmi/Documents/cvusa_cvpr2017/crossnet-master/splits/train_xfork_cvusa.txt',  
   use_GAN=1,
   use_L1=1,

}


-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)



local input_nc = opt.input_nc
local output_nc_seg = opt.output_nc_seg
local output_nc = opt.output_nc

-- translation direction
local idx_ground = nil
local idx_aerial = nil

idx_aerial = {1, input_nc}
idx_ground = {input_nc+1, input_nc+output_nc}

if opt.display == 0 then opt.display = false end



opt.manualSeed = torch.random(1, 10000) -- fix seed
-- opt.manualSeed = 227 -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')



----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end




local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 0.9
local fake_label = 0

local edgemap_nc = 1

------------------------------------------------------------------------------------





function defineG(input_nc, output_nc_seg, output_nc, ngf)
    local netG = nil

    print (input_nc)
    print (output_nc_seg)
    print (output_nc)
    print (ngf)

    if  opt.which_model_netG == "forkG" then netG = defineG_encoder_decoder_fork(input_nc, output_nc_seg, output_nc, ngf)
    elseif  opt.which_model_netG == "forkG_cvusa" then netG = defineG_encoder_decoder_fork_cvusa(input_nc+edgemap_nc, output_nc_seg, output_nc, ngf)
    elseif  opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    else    error("unsupported netG model")
    end

    netG:apply(weights_init)
    return netG
end

function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc + edgemap_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels
    end
    
    if     opt.which_model_netD == "basic_D" then netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
    else error("unsupported netD model")
    end
    
    netD:apply(weights_init)
    
    return netD
end

---------------------------------------------------------------------------

-- load saved models and finetune
if opt.continue_train == 1 then
   print('loading previously trained netG...')
   netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, opt.which_epoch .. '_net_G.t7'), opt)
   print('loading previously trained netD...')
   netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, opt.which_epoch .. '_net_D.t7'), opt)
else
  print('define model netG...')
  netG =  defineG(input_nc, output_nc_seg, output_nc, ngf)
  print('define model netD...')
  netD = defineD(input_nc, output_nc, ndf)
end


print(netG)
print(netD)


local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()


optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta,
}

optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta,
}
---------------------------------------------------------------------------

----------------------------------------------------------------------------
local real_A = torch.Tensor(opt.batchSize, input_nc + edgemap_nc, 256, 1024)    -- real image view 2
local real_B = torch.Tensor(opt.batchSize, input_nc, 512, 512)     -- real image view 1
local fake_B = torch.Tensor(opt.batchSize, output_nc, 512, 512)    -- real image view 2



local real_Bs = torch.Tensor(opt.batchSize, output_nc, 512, 512)    -- real image view 2
local fake_Bs = torch.Tensor(opt.batchSize, output_nc, 512, 512)    -- real image view 2



opt.condition_GAN = 0

local fake_img = torch.Tensor(opt.batchSize, output_nc + output_nc_seg, 512, 512)


local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN + edgemap_nc, 512, 512) -- real image pairs in two views for D
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN + edgemap_nc, 512, 512) -- image pairs in two views for D, one is synthesized



local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()



if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   real_A = real_A:cuda();
   real_B = real_B:cuda(); 
   fake_B = fake_B:cuda();
   fake_Bs = fake_Bs:cuda();
   
   real_Bs = real_Bs:cuda(); 
   real_AB = real_AB:cuda(); 
   fake_AB = fake_AB:cuda();

   if opt.cudnn==1 then
      netG = util.cudnn(netG); 
      netD = util.cudnn(netD);
   end
   criterion:cuda(); criterionAE:cuda();
   netD:cuda(); 
   netG:cuda();  
   print('done')
else
  print('running model on CPU')
end


-- parameters for network
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()



function load_and_preprocess_regular_image(pathA, pathB, pathBs)
  local imA = image.load(pathA, 3, 'float')
  local imB = image.load(pathB, 3, 'float')
  local imBs = image.load(pathBs, 3, 'float')
  
  local perm = torch.LongTensor{3, 2, 1}
  
  imA = imA:index(1, perm)--:mul(256.0): brg, rgb
  imA = imA:mul(2):add(-1)
  assert(imA:max()<=1,"A: badly scaled inputs")
  assert(imA:min()>=-1,"A: badly scaled inputs")
  
  imB = imB:index(1, perm)--:mul(256.0): brg, rgb
  imB = imB:mul(2):add(-1)
  assert(imB:max()<=1,"A: badly scaled inputs")
  assert(imB:min()>=-1,"A: badly scaled inputs")
  
  imBs = imBs:index(1, perm)--:mul(256.0): brg, rgb
  imBs = imBs:mul(2):add(-1)
  assert(imBs:max()<=1,"A: badly scaled inputs")
  assert(imBs:min()>=-1,"A: badly scaled inputs")
  
  
  return imA, imB, imBs
end


function deprocess(img)
    -- BGR to RGB
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm)
    -- [-1,1] to [0,1]
    img = img:add(1):div(2)
    return img
end

----------------------------------------------------------------------------------------------------------------
batchSize = opt.batchSize

m = csvigo.load({path = opt.train_data_path, mode = "large"})
local number_of_files = #m
if batchSize > number_of_files then batchSize = number_of_files end
num_files = math.floor(number_of_files/batchSize) - 1
number_of_files = num_files * batchSize
--number_of_files = 20
print(number_of_files)
print(num_files)
print(batchSize)
local edges = torch.Tensor(edgemap_nc, 256, 1024)

-- create positive and negative examples/pairs
--------------------------------------------------------------------------------------------------------------
function createImageBatch(n)
  data_tm:reset(); data_tm:resume()
  for i = 1, batchSize do
    local aerial_image, ground_image, aerial_seg = load_and_preprocess_regular_image(opt.DATA_ROOT .. '/' .. m[perm_rand[n+i-1]][1], opt.DATA_ROOT .. '/' .. m[perm_rand[n+i-1]][2], opt.DATA_ROOT .. '/' .. m[perm_rand[n+i-1]][3])
    aerial_image = image.scale(aerial_image, 512, 512)
    ground_image = image.scale(ground_image, 1024, 256)
    aerial_seg = image.scale(aerial_seg, 512, 512)

    gray_ground_image = image.load(opt.DATA_ROOT .. '/' .. m[perm_rand[n+i-1]][2], 1, 'byte')   -- [ 1 in the parameter returns image of 1 channel , i.e. grayscale]
    gray_ground_image = image.scale(gray_ground_image, 1024, 256)

    edges[1] = (cv.Canny{image=gray_ground_image, threshold1=50, threshold2=100})/255
    
    real_A[i] = torch.cat(ground_image, edges, 1)

    real_B[i] = aerial_image
    real_Bs[i] = aerial_seg
  end
  data_tm:stop()

  -- create fake
  fake_img = netG:forward(real_A)
  fake_B = fake_img[1]
  fake_Bs = fake_img[2]

  real_AB = real_B -- unconditional GAN, only penalizes structure in B
  fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
  -- end
end
----------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

    gradParametersD:zero()
    
    -- Real
    local output = netD:forward(real_AB)
    local label = torch.FloatTensor(output:size()):fill(real_label)
    
    if opt.gpu>0 then 
      label = label:cuda()
    end
    
    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(real_AB, df_do)   
   
   
    -- Fake
    local output = netD:forward(fake_AB)
    label:fill(fake_label)
    local errD_fake = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(fake_AB, df_do)
    
    
    errD = (errD_real + errD_fake)/2
    
    return errD, gradParametersD
end

---- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()
    
    -- GAN loss
    local df_dg = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
      df_dg = df_dg:cuda();
    end
    
    if opt.use_GAN==1 then
       local output = netD:forward(fake_AB)
       local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
       if opt.gpu>0 then 
        label = label:cuda();
       end
       
       errG = criterion:forward(output, label)
       local df_do = criterion:backward(output, label)
       df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
    else
        errG = 0
    end
    
    -- unary loss
    local df_do_AE = torch.zeros(fake_B:size())
    local a = torch.zeros(fake_B:size())
    local df_do_AE_seg = torch.zeros(fake_Bs:size())
    local b = torch.zeros(fake_Bs:size())

    if opt.gpu>0 then 
      df_do_AE = df_do_AE:cuda();
      a = a:cuda();
      df_do_AE_seg = df_do_AE_seg:cuda();
      b = b:cuda()
    end
    if opt.use_L1==1 then
       errL1 = criterionAE:forward(fake_B, real_B)
       df_do_AE = criterionAE:backward(fake_B, real_B)
       a = df_dg + df_do_AE:mul(opt.lambda1)
       
       errL1_seg = criterionAE:forward(fake_Bs, real_Bs)
       df_do_AE_seg = criterionAE:backward(fake_Bs, real_Bs)
       b = df_do_AE_seg:mul(opt.lambda2)

    else
        errL1 = 0
        a = df_dg + df_do_AE:mul(opt.lambda1)
        errL1_seg = 0
        b = df_do_AE_seg:mul(opt.lambda2)
    end
    
    local s ={a,b}
    netG:backward(real_A, s)
    
    return errG, gradParametersG
end


----------------------------------------------------------------------------------------------------------------

-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()
local counter = 0


for epoch = 1 + opt.which_epoch, opt.niter do
  collectgarbage()
    epoch_tm:reset()
    perm_rand = torch.randperm(number_of_files)
    for i = 1, number_of_files, opt.batchSize do
      collectgarbage()
        tm:reset()


        -- load a batch and train on that batch
        createImageBatch(i)
        
        -- (1) Update D1 network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end
                
        -- (2) Update G1 network: maximize log(D(x,G(x))) + L1(y,G(x))
        optim.adam(fGx, parametersG, optimStateG)
       
        -- display
        counter = counter + 1

        -- logging
        if counter % opt.print_freq == 0 then
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f  ErrL1_seg: %.4f'):format(
                     epoch, ((i-1) / opt.batchSize),
                     math.floor(math.min(number_of_files, opt.ntrain) / opt.batchSize),
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errG and errG or -1, errD and errD or -1, errL1 and errL1 or -1, errL1_seg and errL1_seg or -1))
            
            
        end
        
        -- save latest model
        if counter % opt.save_latest_freq == 0 then
            print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
        end
        
    end

    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil

    if epoch % opt.save_epoch_freq == 0 then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())

    end
    
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, opt.niter, epoch_tm:time().real))
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
    collectgarbage()

end
