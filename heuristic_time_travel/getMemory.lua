
require('paths')
require('nngraph')
require('cunn')
require('optim')
paths.dofile('params.lua')
paths.dofile('utils.lua')
paths.dofile('layers/Normalization.lua')
torch.setdefaulttensortype('torch.FloatTensor')
g_make_deterministic(123)

-- load the data
--trdata = paths.dofile('data.lua')
--tedata = paths.dofile('data.lua')
--trdata:load(trfiles, opt.batchsize, opt.T)
--tedata:load(tefiles, opt.batchsize, trdata.memsize, trdata.dict, trdata.idict)

-- split into train and validation sets
--nquestions = trdata.questions:size(1)
--train_range = torch.range(1, math.floor(0.9*nquestions))
--val_range = torch.range(math.floor(0.9*nquestions) + 1, nquestions)

-- set some parameters based on the dataset
--opt.nwords = #trdata.idict
--opt.winsize = trdata.memory:size(3)
--if opt.tied == 1 then 
  -- opt.memslots = trdata.entities:size(1)
   --print('tying keys to entities -> ' .. opt.memslots .. ' memory slots')
--end


-- build the model and loss
paths.dofile(opt.model .. '.lua')
save_file = 'outputs-bow/task=21-model=entnet-embed=bow-edim=100-initstd=0.1-sdt_decay=25-optim=adam-sdt=0.01-nhop=1-tied=1-T=70_init_num_0.model'
print("Save file is")
print(save_file)
obj = torch.load(save_file)
model = obj['model']
print("model loaded")


function evaluate(split, display)
   if opt.dropout > 0 then 
      model:setDropout('test')
   end
    local total_err, total_cost, total_num = 0, 0, 0
    local N, indx
    if split == 'train' then
        N = train_range:size(1)
        indx = train_range
        data = trdata
    elseif split == 'valid' then
        N = val_range:size(1)
        indx = val_range
        data = trdata
    elseif split == 'test' then
        N = tedata.questions:size(1)
        indx = torch.range(1, N)
        data = tedata
    end
    local loss = torch.Tensor(N)
    print("inside this loop")
    for k = 1, math.floor(N/opt.batchsize) do
        local batch = indx:index(1, torch.range(1 + (k-1)*opt.batchsize, k*opt.batchsize):long())
        local question, answer, story, facts, graph = data:getBatch(batch)
        local err, cost, missed = model:fprop(question, answer, story, graph)
        total_cost = total_cost + cost
        total_err = total_err + err
        total_num = total_num + opt.batchsize
    end
    return total_err / total_num
 end


print(model.network)
graph.dot(model.network.fg,'graph','mygraph')
--evaluate('test')
