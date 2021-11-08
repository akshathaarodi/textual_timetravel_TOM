--------------------------
-- create the EntNet
--------------------------
--nngraph.setDebug(true)
local vocabsize = opt.nwords
if opt.tied == 0 then
   -- add extra words to the vocabulary representing keys
   vocabsize = vocabsize + opt.memslots
end

-- function to take a set of word embeddings and produce a fixed-size vector
function input_encoder(opt, input, model, label)
   local input = nn.View(-1, opt.winsize * opt.edim)(input)
    if model == 'bow' then
       return nn.Sum(2)(nn.View(opt.batchsize, opt.winsize, opt.edim)(input))
    elseif model == 'icmul+bow' then
       input = nn.Dropout(opt.dropout)(input):annotate{name = 'dropout'}
       input = nn.View(-1, opt.winsize * opt.edim)(input)
       input = nn.CMul(opt.winsize * opt.edim)(input):annotate{name = label}
       return nn.Sum(2)(nn.View(opt.batchsize, opt.winsize, opt.edim)(input))
    else
       error('encoder not recognized')
    end
end

-- output layer
local function output_module(opt, hops, x, M, keyvecs)
   local hid = {}
   local s = nn.LookupTable(vocabsize, opt.edim)(x):annotate{name = 'E'}
   hid[0] = input_encoder(opt, s, opt.embed, 'q_embed1'):annotate{name = 'q'}
   local keys = nn.Replicate(opt.batchsize, 1, 3)(nn.View(1, opt.memslots, opt.edim)(keyvecs))

--   keys  = nn.View(opt.batchsize * opt.memslots, opt.edim)(nn.Contiguous()(keys))
--   Mkeys = nn.CAddTable()({M,keys})
   keys = nn.View(opt.batchsize, opt.memslots, opt.edim)(nn.Contiguous()(keys)):annotate{name = "out_keys"}
   for h = 1, hops do
      local hid3dim = nn.View(opt.batchsize, 1, opt.edim)(hid[h-1])
      local MM1aout = nn.MM(false, true)
      local MM2aout = nn.MM(false, true)
      local Aout = MM1aout({hid3dim, M})
      local A2out = MM2aout({hid3dim, keys})
      local Aout2dim = nn.View(opt.batchsize, -1)(Aout)
      local A2out2dim = nn.View(opt.batchsize,-1)(A2out)
      local P = nn.SoftMax()(Aout2dim):annotate{name = 'predictions'}
      local P2 = nn.SoftMax()(A2out2dim):annotate{name = 'predictions2'}
      local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)--:annotate{name = "P2"}
      local probs23dim = nn.View(1, -1):setNumInputDims(1)(P2)--:annotate{name = "P2"}
      local MM1bout = nn.MM(false, false)
      local MM2bout = nn.MM(false, false)
      local Bout = nn.View(opt.batchsize, -1)(MM1bout({probs3dim, M}))
      local B2out = nn.View(opt.batchsize, -1)(MM2bout({probs23dim, keys}))
      local C = nn.Linear(opt.edim, opt.edim, false)(Bout):annotate{name = 'H'}
      local C2 = nn.Linear(opt.edim, opt.edim, false)(B2out):annotate{name = 'H2'}
--      keys = nn.Squeeze()(keys):annotate{name = "keys"} -- 32 * 8 * 100
--      local memm = nn.Replicate(opt.memslots,2)(nn.Unsqueeze(2)(hid[h-1]))
--      C = nn.Replicate(opt.memslots)(nn.Unsqueeze(2)(C))
      local D = nn.CAddTable()({hid[h-1], C})
      local D2 = nn.CAddTable()({D, C2})
      hid[h] = nn.PReLU(opt.edim)(D2):annotate{name = 'prelu'}
   end
   local z = nn.Linear(opt.edim, vocabsize, false)(hid[hops]):annotate{name = 'z'}
   local pred = nn.LogSoftMax()(nn.Narrow(2, 1, opt.nwords)(z))
   return pred
end


function update_mem_from_gates(gate, opt, keys, sentence, memories, t, mask)
   gate = nn.Replicate(opt.edim, 2, 2)(nn.View(opt.batchsize * opt.memslots, 1)(gate))

   -- compute candidate memories
   local U = nn.Linear(opt.edim, opt.edim)(memories):annotate{name = 'U'}
   local V = nn.Linear(opt.edim, opt.edim, false)(sentence):annotate{name = 'V'}
   local W = nn.Linear(opt.edim, opt.edim, false)(keys):annotate{name = 'W'}
   local candidate_memories = nn.PReLU(opt.edim)(nn.CAddTable(){U, V, W}):annotate{name = 'prelu'}

   -- update the memories
   local updated_memories = nn.CAddTable(){memories, nn.CMulTable(){gate, candidate_memories}}
   --print("Early return gare")
   --print(return_gate)
   -- normalize to the sphere
   updated_memories = nn.Normalization()(updated_memories)
   return nn.View(opt.batchsize, opt.memslots, opt.edim)(updated_memories)
end

-- dynamic memory layer
local function update_memory(opt, keys, sentence, memories, t, mask)
   -- reshape everything to 2D so it can be fed to nn.Linear
   local sentence = nn.Replicate(opt.memslots, 2, 3)(nn.View(opt.batchsize, 1, opt.edim)(sentence))
   sentence       = nn.View(opt.batchsize * opt.memslots, opt.edim)(nn.Contiguous()(sentence))
   local keys     = nn.Replicate(opt.batchsize, 1, 3)(nn.View(1, opt.memslots, opt.edim)(keys)):annotate{name='keys'}
   keys           = nn.View(opt.batchsize * opt.memslots, opt.edim)(nn.Contiguous()(keys))
   local mask     = nn.Replicate(opt.memslots, 2, 2)(mask)
   local memories = nn.View(opt.batchsize * opt.memslots, opt.edim)(memories)

   local function DotBias(a, b)
      return nn.Add(opt.memslots)(nn.View(opt.batchsize, opt.memslots)(nn.DotProduct(){a, b})):annotate{name = 'gate_bias'}
   end

   -- compute the gate activations (mask indicates end of story which forces gates to close)
   local gate = nn.Sigmoid()(DotBias(nn.CAddTable(){memories, keys}, sentence))
   gate = nn.CMulTable(){gate, mask}:annotate{name = 'gate' .. t}
   local return_gate = gate
   gate = nn.Replicate(opt.edim, 2, 2)(nn.View(opt.batchsize * opt.memslots, 1)(gate))

   -- compute candidate memories
   local U = nn.Linear(opt.edim, opt.edim)(memories):annotate{name = 'U'}
   local V = nn.Linear(opt.edim, opt.edim, false)(sentence):annotate{name = 'V'}
   local W = nn.Linear(opt.edim, opt.edim, false)(keys):annotate{name = 'W'}
   local candidate_memories = nn.PReLU(opt.edim)(nn.CAddTable(){U, V, W}):annotate{name = 'prelu'}

   -- update the memories
   local updated_memories = nn.CAddTable(){memories, nn.CMulTable(){gate, candidate_memories}}
   --print("Early return gare")
   --print(return_gate)
   -- normalize to the sphere
   updated_memories = nn.Normalization()(updated_memories)
   return nn.View(opt.batchsize, opt.memslots, opt.edim)(updated_memories)
end


local function get_gates(opt, keys, sentence, memories, t, mask)
   -- reshape everything to 2D so it can be fed to nn.Linear
   local sentence = nn.Replicate(opt.memslots, 2, 3)(nn.View(opt.batchsize, 1, opt.edim)(sentence))
   sentence       = nn.View(opt.batchsize * opt.memslots, opt.edim)(nn.Contiguous()(sentence))
   local keys     = nn.Replicate(opt.batchsize, 1, 3)(nn.View(1, opt.memslots, opt.edim)(keys)):annotate{name='keys'}
   keys           = nn.View(opt.batchsize * opt.memslots, opt.edim)(nn.Contiguous()(keys))
   local mask     = nn.Replicate(opt.memslots, 2, 2)(mask)
   local memories = nn.View(opt.batchsize * opt.memslots, opt.edim)(memories)

   local function DotBias(a, b)
      return nn.Add(opt.memslots)(nn.View(opt.batchsize, opt.memslots)(nn.DotProduct(){a, b})):annotate{name = 'gate_bias'}
   end

   -- compute the gate activations (mask indicates end of story which forces gates to close)
   local gate = nn.Sigmoid()(DotBias(nn.CAddTable(){memories, keys}, sentence))
   gate = nn.CMulTable(){gate, mask}:annotate{name = 'gate' .. t}
   return gate
end

-- split update memory into 2 and try to return and see
local function get_gates_memory(opt, keys, sentence, memories, t, mask)
   --print("t is")
   --print(t)
   gate = get_gates(opt, keys, sentence, memories, t, mask)
   memory = update_mem_from_gates(gate, opt, keys, sentence, memories, t, mask)
   return gate, memory
end
-- build the nngraph module
local function build_network(opt)
   local question = nn.Identity()()
   local story    = nn.Identity()()
   local keys     = nn.Identity()()
   local mask     = nn.Identity()()
   local memories = {}

   local initmems = nn.Replicate(opt.batchsize, 1, 2)(keys)
   memories[0]    = nn.LookupTable(vocabsize, opt.edim)(initmems):annotate{name = 'E'}
   local keyvecs  = nn.LookupTable(vocabsize, opt.edim)(keys):annotate{name = 'E'}
   --local return_gates = {}
   --local gates_tensor = torch.Tensor(opt.T,opt.batchsize,opt.memslots):cuda()
   for i = 1, opt.T do
      local sentence = input_encoder(opt, nn.LookupTable(vocabsize, opt.edim)(nn.Select(2, i)(story)):annotate{name = 'E'}, opt.embed, 's_embed1')
      --memories[i], return_gates[i]    = get_gates_memory(opt, keyvecs, sentence,  memories[i - 1], i, nn.Select(2, i)(mask))
      memories[i] = update_memory(opt, keyvecs, sentence,  memories[i - 1], i, nn.Select(2, i)(mask)):annotate{name = 'memo'..i}
      --print("Memories are")
      --print(memories[i])
      --return_gates[i] = get_gates(opt, keyvecs, sentence,  memories[i - 1], i, nn.Select(2, i)(mask))
   end
   --return_gates = return_gates:transpose(1,2)
   local pred = output_module(opt, opt.nhop, question, memories[opt.T], keyvecs)
   --[[
   local gates_tensor = torch.Tensor(opt.T,opt.batchsize,opt.memslots)
   for i=1, opt.T do
          --pred_gates_tensor[i] = getNodesByName("gate"..i).output
      gates_tensor[i] = return_gates[i][0]
   end]]--
   --local gate_tensor = torch.Tensor(return_gates)
   --local gates = nn.Identity()(return_gates)--(gates_tensor:transpose(1,2))
   return nn.gModule({question, story, keys, mask}, {pred})
end

-- build the final model
function build_model(opt)
   local model = {}
   model.network = build_network(opt)
   model.network = model.network:cuda()

   if opt.tied == 0 then
      model.keys = torch.range(opt.nwords + 1, opt.nwords + opt.memslots)
   else
      model.keys = trdata.entities
   end
   model.keys = model.keys:cuda()

   -- share the clones across timesteps
   share_modules({get_module(model.network, 'prelu')})
   share_modules({get_module(model.network, 'gate_bias')})
   share_modules({get_module(model.network, 'U')})
   share_modules({get_module(model.network, 'V')})
   share_modules({get_module(model.network, 'W')})
   share_modules({get_module(model.network, 'q_embed1')})
   share_modules({get_module(model.network, 's_embed1')})
   share_modules({get_module(model.network, 'E')})
   share_modules({get_module(model.network, 'H')})

   model.paramx, model.paramdx = model.network:getParameters()
   model.loss = nn.ClassNLLCriterion():cuda()
   model.loss.sizeAverage = false
   --model.auxloss = nn.BCECriterion():cuda()
   --model.auxloss.sizeAverage = false
   -- do not average the loss, just sum up

   function model:reset()
      -- initialize weight to a Gaussian
      self.paramx:normal(0, opt.init_std)

      -- initialize PReLU slopes to 1
      local prelus = get_module(self.network, 'prelu')
      for i = 1, #prelus do
         prelus[i].weight:fill(1)
      end

      -- initialize encoder mask weights to 1 (i.e. BoW)
      if opt.embed == 'icmul+bow' then
         local icmul = get_module(self.network, 'q_embed1')
         for i = 1, #icmul do
            local w = icmul[i].weight
            w:fill(1)
         end
      end
   end

   function model:zeroNilToken()
      local G = get_module(self.network, 'E')
      local Z = get_module(self.network, 'z')
      for i = 1, #G do G[i].weight[1]:zero() end
      for i = 1, #Z do Z[i].weight[1]:zero() end
   end

   function model:setDropout(split)
      local drop = get_module(self.network, 'dropout')
      for i = 1, #drop do
         drop[i].train = (split == 'train')
      end
   end

   function model:fprop_cpu(question, answer, story)
      self.mask = story:ne(1):sum(3):select(3,1):ne(0)
      self.logprob = self.network:forward({question:float(), story:float(), self.keys:float(), self.mask:float()})
      local cost = self.loss:forward(self.logprob, answer:float())
      local _, pred = self.logprob:max(2)
      local missed = pred:ne(answer:long())
      return missed:sum(), cost, missed, pred
   end


   function model:fprop(question, answer, story)
      self.mask = story:ne(1):sum(3):select(3,1):ne(0):cuda()
      --pred_gates = {}
      --print("in fprop")
      --self.logprob
      self.logprob = self.network:forward({question, story, self.keys, self.mask})
      --print("out is")
      --print(out)
      --print("end out")
      local cost = self.loss:forward(self.logprob, answer)
      local _, pred = self.logprob:max(2)
      pred = pred:cuda()
      local missed = pred:ne(answer)
      return missed:sum(), cost, missed, pred, self.keys
   end

   function model:bprop(question, answer, story)
      self.network:zeroGradParameters()
      -- gradInput = derivative of the cost w.r.t layer's input
      --gradOutput = derivative of the cost w.r.t layer's output
      -- gradInput = BCECriterion:updateGradInput(input, target)
      local grad = self.loss:updateGradInput(self.logprob, answer)
      self.network:backward({question, story, self.keys, self.mask}, grad)
      local gradnorm = self.paramdx:norm()
      if gradnorm > opt.maxgradnorm then
         self.paramdx:mul(opt.maxgradnorm / gradnorm)
      end
      self:zeroNilToken()
   end

   return model
end

function Set (list)
  local set = {}
  --print("list is")
  --print(list)
  for i=1, opt.memslots do set[list[i]] = true end
  return set
end

-- t in the tensor that is storing the entities in each sentence
-- k is the kth sentence of the stoyr
-- word is the one we are looking for over lap
function get_combined_entities(entities_table, entities)
    -- create a shallow copy of entities table
    local entities_table_comb = {}
    for key,v in pairs(entities) do    
        entities_table_comb[key] = true
        for i = 1, #entities_table-1 do
            if entities_table[i][key] then
                 for k,_ in pairs(entities_table[i]) do
                      entities_table_comb[k]=true
                  end
             end
         end
     end
     return entities_table_comb
end

function get_sent_target(entities, keys)
    target_gate = torch.Tensor(opt.memslots):zero()
    for i=1, opt.memslots do
         if entities[keys[i]] then
             target_gate[i] = 1
         end
    end
    return target_gate
end
    
function get_target_gates(keys, story)
   key_set = Set(keys)
-- story is 32 * 70 * 10
   target_gates = torch.Tensor(opt.batchsize, opt.T, opt.memslots)
   all_story_gates = {}
   for i = 1, opt.batchsize do
      indv_story = story[i]
      --print("Indv story is")
      --print(indv_story)
      -- 70 * 10
      -- stores all the entities in the story per sentencec
      entities_table = {}
      combined_entities_table = {}
      -- stores target tensors per sentence
      target_gates_sent = torch.Tensor(opt.T, opt.memslots)
      for j=1, opt.T do
           sentence = indv_story[j]
           entities = {}
           for k=1, 10 do
               word = sentence[k]
               if key_set[word] then
                   entities[word] = true
               end
           end
           table.insert(entities_table,entities)
           combined_entities_table[j] = get_combined_entities(entities_table, entities)
           target_gates_sent[j] = get_sent_target(combined_entities_table[j], keys)
      end
      table.insert(all_story_gates, target_gates_sent)
      target_gates[i] = target_gates_sent
    end
    --print("target_gates1 are")
    --print(target_gates[1])
    --print("target_gates1 completed")
    --print("Keys are")
    --print(keys)
    return target_gates
end    


-- http://wiki.roblox.com/index.php/Cloning_tables
function shallowCopy(original)
    local copy = {}
    for key, value in pairs(original) do
        copy[key] = value
    end
    return copy
end

function cpu_copy(model)
    local model_clone = shallowCopy(model)
    -- clone original model params and convert to cpu
    model_clone.network = model.network:clone():float()
    model_clone.paramx, model_clone.paramdx = model.paramx:clone():float(), model.paramdx:clone():float()
    model_clone.keys = model.keys:clone():float()
    model_clone.loss = model.loss:clone():float()
    model_clone.mask = model.mask:clone():float()
    model_clone.logprob = model.logprob:clone():float()
    return model_clone
end

