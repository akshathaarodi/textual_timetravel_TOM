require('paths')
require('nngraph')
require('cunn')
require('optim')
require('visdom')
require('gnuplot')
local nnq = require 'nnquery'
local nninit = require 'nninit'

paths.dofile('params.lua')
paths.dofile('utils.lua')
paths.dofile('layers/Normalization.lua')
torch.setdefaulttensortype('torch.FloatTensor')
g_make_deterministic(123)

-- load the data
trdata = paths.dofile('data.lua')
tedata = paths.dofile('data.lua')
tvdata = paths.dofile('data.lua')
trdata:load(trfiles, opt.batchsize, opt.T)
tedata:load(tefiles, opt.batchsize, trdata.memsize, trdata.dict, trdata.idict)
tvdata:load(tvfiles, opt.batchsize, trdata.memsize, trdata.dict, trdata.idict)

out_folder_name = 'outputs'
os.execute("mkdir "..out_folder_name)
os.execute("cp params.lua "..out_folder_name)

-- split into train and validation sets
nquestions = trdata.questions:size(1)

train_range = torch.range(1, nquestions)
nvalquestions = tvdata.questions:size(1)
val_range = torch.range(1, nvalquestions)
-- set some parameters based on the dataset
opt.nwords = #trdata.idict
opt.winsize = trdata.memory:size(3)
if opt.tied == 1 then 
   opt.memslots = trdata.entities:size(1)
   print('tying keys to entities -> ' .. opt.memslots .. ' memory slots')
end

-- build the model and loss
paths.dofile(opt.model .. '.lua')
model = build_model(opt)


if opt.load_pretrained_model==1 then
   --Load model
   save_name = opt.save..'/final.model'
   obj = torch.load(save_name)
   --print("saved file is")
   --print(obj)
   model = obj['model']
end
print('\nmodel: ' .. paths.basename(opt.modelFilename))
print('#params = ' .. model.paramx:size(1))

function getNodesByName(name)
    c = 0
    for _, node in ipairs(model.network.forwardnodes) do
        if node.data.annotations.name == name then
             c = c +1
             return node.data.module
             --print("printing memo from main"..c)
             --print(node.data.module.output)
             --print("printing memo from main")
        end
    end
    print("Couldnt find for node"..name)
    return nil
end

function train()
    local train_err = {}
    local train_cost = {}
    local val_err = {}
    optstate = {learningRate = opt.sdt}

    for ep = 1, opt.epochs do
        if ep % opt.sdt_decay_step == 0 and opt.sdt_decay_step ~= -1 then
            optstate.learningRate = optstate.learningRate / 2
        end
        model:zeroNilToken()
        if opt.dropout > 0 then 
           model:setDropout('train')
        end
        local total_err, total_cost, total_num = 0, 0, 0
        local nBatches = math.floor(train_range:size(1)/opt.batchsize)
        --print("in train, n batches is"..nBatches..train_range:size(1))
        for k = 1, nBatches do
           xlua.progress(k, nBatches)
            local err, cost
            local feval = function ()
                local batch = train_range:index(1, torch.randperm(train_range:size(1)):sub(1, opt.batchsize):long())
                local question, answer, story = trdata:getBatch(batch)
                err, cost = model:fprop(question, answer, story, graph)
                model:bprop(question, answer, story, sdt)
                return cost, model.paramdx
            end
            optimize(feval, model.paramx, optstate)
            model:zeroNilToken()
            total_cost = total_cost + cost
            total_err = total_err + err
            total_num = total_num + opt.batchsize

            if k % 10 == 0 then 
               collectgarbage()
               collectgarbage()
            end
        end
        train_err[ep] = total_err / total_num
        train_cost[ep] = total_cost / total_num
        val_err[ep] = evaluate('valid')
        --val_err[ep] = evaluate('test')

        local log_string = 'epoch = ' .. ep
            .. ' | train cost = ' .. g_f4(train_cost[ep])
            .. ' | train err = ' .. g_f4(train_err[ep])
            .. ' | valid err = ' .. g_f4(val_err[ep])
            .. ' | lr = ' .. optstate.learningRate

        print(log_string)
        collectgarbage()
    end
    return val_err[opt.epochs], train_err, val_err
end


function get_words(data, sent, q)
   local sent_words = {}
   if q == 0 then
      return data.idict[sent]
   end
   for k = 1, q do
       table.insert(sent_words, data.idict[sent[k]])
   end
   return sent_words
end

key_names = keys

function evaluate(split, display)
   print('Evaluating '..split)
   if opt.dropout > 0 then 
      model:setDropout('test')
   end
    local predictions_file, answers_file, questions_file = '', '', ''
    local total_err, total_cost, total_num = 0, 0, 0
    local N, indx
    if split == 'train' then
        N = train_range:size(1)
        indx = torch.range(1, N)
        data = trdata
    elseif split == 'valid' then --unused
        N = val_range:size(1)
        indx = val_range
        data = tvdata

        predictions_file = io.open(out_folder_name.."/predictions.txt", "w")
        answers_file = io.open(out_folder_name.."/answers.txt","w")
        questions_file = io.open(out_folder_name.."/questions.txt","w")


    elseif split == 'test' then
        N = tedata.questions:size(1)
        indx = torch.range(1, N)
        data = tedata

        predictions_file = io.open(out_folder_name.."/predictions_test.txt", "w")
        answers_file = io.open(out_folder_name.."/answers_test.txt","w")
        questions_file = io.open(out_folder_name.."/questions_test.txt","w")
    
    elseif split == 'unit-test' then
        N = tedata.questions:size(1)
        indx = torch.range(1, N)
        data = tudata
        print("Running unit test")
    end
    local loss = torch.Tensor(N)
    for k = 1, math.floor(N/opt.batchsize) do
        xlua.progress(k, math.floor(N/opt.batchsize))
        local batch = indx:index(1, torch.range(1 + (k-1)*opt.batchsize, k*opt.batchsize):long())
        local question, answer, story, facts, graph = data:getBatch(batch)
        local err, cost, missed, pred, keys = model:fprop(question, answer, story, graph)
        key_names = keys
        --print("err is"..err)
        if split == 'valid' then
            for p = 1, opt.batchsize do
                a = answer[p]
                answers_file:write(get_words(data,a,0)..'\n')
                predictions_file:write(get_words(data,pred[p], 1)[1]..'\n')
                q = get_words(data, question[p], 10)
                qs = '' 
                for i=1, #q do
                     qs = qs..q[i]..' '
                end
                questions_file:write(qs..'\n')
            end
        end


        if split == 'test' then
            for p = 1, opt.batchsize do
                a = answer[p]
                answers_file:write(get_words(data,a,0)..'\n')
                predictions_file:write(get_words(data,pred[p], 1)[1]..'\n')
                q = get_words(data, question[p], 10)
                qs = '' 
                for i=1, #q do
                     qs = qs..q[i]..' '
                end
                questions_file:write(qs..'\n')
            end
        end

        total_cost = total_cost + cost
        total_err = total_err + err
        total_num = total_num + opt.batchsize
    end
    if split == 'test' or split == 'valid' then
       predictions_file:close()
       questions_file:close()
       answers_file:close()
    end
    return total_err / total_num
 end

final_perf_train = {}
final_perf_val = {}
final_perf_test = {}
final_perf_utest = {}
--weights = {}
if opt.load_pretrained_model ==0 then
    for i = 1, opt.runs do
        print('--------------------')
        print('RUN ' .. i)
        print('--------------------')
    -- reset the weights 
        g_make_deterministic(opt.save_num)
        model:reset()
    -- train
    

        final_perf_val[i] = train()
        final_perf_train[i] = evaluate('train')
        final_perf_test[i] = evaluate('test')
        print('val err')
        print(final_perf_val)
        print('test err')
        print(final_perf_test)

        if opt.save ~= '' then
           local log_string = 'run ' .. opt.save_num 
              .. ' | train error = ' .. g_f4(final_perf_train[i]) 
              .. ' | valid error = ' .. g_f4(final_perf_val[i]) 
              .. ' | test error = ' .. g_f4(final_perf_test[i])
           write(opt.modelFilename .. '.log', log_string)
           save_file = opt.modelFilename .. '_init_num_' .. opt.save_num .. '.model' 
        end
        if final_perf_val[i] == 0 then
       -- we will pick this run and don't need more
           print("Ending training at epoch"..i)
           break
        end
    end
end



function getKeysSortedByValue(tbl, sortFunction)
  local keys = {}
  for key in pairs(tbl) do
    table.insert(keys, key)
  end

  table.sort(keys, function(a, b)
    return sortFunction(tbl[a], tbl[b])
  end)

  return keys
end

function spairs(t, order)
    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys 
    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end

function sortAttention(tbl)
    local sortedtbl = {}
    for k,v in spairs(tbl, function(t,a,b) return t[b] < t[a] end) do
       print(k,v)
    end
    return sortedtbl
end

function sortAttention_file(tbl, file)
    local sortedtbl = {}
    for k,v in spairs(tbl, function(t,a,b) return t[b] < t[a] end) do
       file:write(k ..' ' .. v..'\n')
    end
    return sortedtbl
end

function get_all_active_keys()
    local gates = {}

    for i=1, opt.T do
        gates[i] = getNodesByName("gate"..i).output[1]
    end
    active_keys_t = {}
    for i=1, opt.T do
        gate = gates[i]
        active_keys = {}
        for j=1, opt.memslots do
            if gate[j]>0.6 then
                active_keys[get_words(data,key_names[j],0)] = gate[j]
            end
        end
        local new_k = sortAttention(active_keys)
        table.insert(active_keys_t, new_k)
    end
end


function run_example_and_plot(split, question_num, out_folder_name, timestep)
    local batch = torch.FloatTensor(32):fill(question_num)
    local num_sents = 6
    local sample_data = nil
    if split == 'valid' then
	    sample_data = tvdata
    elseif split == 'test' then
	    sample_data = tedata
    end

    file_name = out_folder_name.."/ex_pred_"..split..'_'..question_num..".txt"
    local ex_prediction = io.open(file_name, "w")
    local example_num = 1
    local question, answer, story, facts, graph = sample_data:getBatch(batch)
    -- story is 32 by 70 by 10
    one_story = story[example_num]
    -- Lets take first 6 sentence of the story
    local sents = {}
    local err, cost, missed, pred, key_names = model:fprop(question, answer, story, graph, timestep)
    local real_keynames = {}

    for i=1, opt.memslots do
        if opt.tied == 1 then
             real_keynames[i] = get_words(sample_data, key_names[i],0)
        else
             real_keynames[i] = key_names[i]
        end
    end
    xtic_label = "set xtics (" 
    for i =1,num_sents do
        local words = get_words(sample_data, one_story[i], 10)
        local sent = ""
        for j =1, 10 do
            if words[j] ~= "" then
                sent = sent.." ".. words[j]
            end
        end
        xtic_label = xtic_label.."\""..sent.."\"".." "..(i-1)..","
    end
    xtic_label = xtic_label..")"
    ytic_label = "set ytics ("

    
    for i=1,opt.memslots do
        ytic_label = ytic_label.."\""..real_keynames[i].."\"".." "..(i-1)..","
    end

    ytic_label = ytic_label..")"
    --get gate outputs
    --print("Question, Answer and Prediction are")
    question_words = get_words(sample_data, question[example_num], 10)
    qs = ''
    for i=1, #question_words do
        qs = qs..question_words[i]..' '
    end
    ex_prediction:write("Question, Answer and Prediction are\n")
    ex_prediction:write(qs.."\n")
    ex_prediction:write(get_words(sample_data,answer[example_num],0).."\n")
    ex_prediction:write(get_words(sample_data,pred[example_num], 1)[1].."\n")
    local gates = {}
    for i=1, opt.T do
        gates[i] = getNodesByName("gate"..i).output[example_num]
    end
    prediction_distribution = getNodesByName("predictions").output[example_num]
    xtic_label_prob = "set xtics ( \""..qs.."\" 0)"

    local prediction_kv ={}
    if opt.tied == 1 then
        for i=1, opt.memslots do
            prediction_kv[get_words(sample_data, key_names[i],0)] = prediction_distribution[i]
        end
    else
        for i=1, opt.memslots do
            prediction_kv[i] = prediction_distribution[i]
        end
    end

    for k,v in spairs(prediction_kv, function(t,a,b) return t[b] < t[a] end) do
        ex_prediction:write(k..'  '..v.."\n")
    end
    

    ex_prediction:close()
    local gate_tensor = torch.Tensor(6,opt.memslots):cuda()

    for i=1,num_sents do
        gate_tensor[i] = gates[i]
    end

    
    --print(gate_tensor)
    local graph_data = gate_tensor:t()

    -- set graph params
    gnuplot.pngfigure(out_folder_name..'/graph_'..split..'_'..question_num..'.png')
    gnuplot.raw('set pm3d map')
    gnuplot.raw('set terminal png size 2500,3500')
    gnuplot.xlabel('Sentences')
    gnuplot.ylabel('Memslots')
    gnuplot.raw(ytic_label)
    gnuplot.raw(xtic_label)
    gnuplot.raw("set ytics scale major")
    gnuplot.raw("set xtics scale major")
    gnuplot.axis('auto')
    -- Uncomment this to plot

    gnuplot.imagesc(graph_data, 'rgbformula -7,-7,2')

    gnuplot.plotflush()

    -- Plot probability distribution
    gnuplot.pngfigure(out_folder_name..'/prob_graph_'..question_num..'.png')
    gnuplot.raw('set pm3d map')
    gnuplot.raw('set terminal png size 2500,3500')
    gnuplot.xlabel('Questions')
    gnuplot.ylabel('Memslots')
    gnuplot.raw(ytic_label)
    gnuplot.raw(xtic_label_prob)
    gnuplot.raw("set ytics scale major")
    gnuplot.raw("set xtics scale major")
    gnuplot.axis('auto')
    local pred_dist_data = torch.expand(prediction_distribution:resize(opt.memslots, 1), opt.memslots, 2)
    gnuplot.imagesc(pred_dist_data, 'rgbformula -7,2,-7')

    gnuplot.plotflush()
    return real_keynames
end


function save_model()
    print("Save model")
    --local log_string = 'run ' .. i 
    --      .. ' | train error = ' .. g_f4(final_perf_train[i]) 
    --      .. ' | valid error = ' .. g_f4(final_perf_val[i]) 
    --      .. ' | test error = ' .. g_f4(final_perf_test[i])
    --write(opt.modelFilename .. '.log', log_string)
    torch.save(opt.modelFilename .. '.model', {final_perf_val = final_perf_val, 
                     final_perf_test = final_perf_test, 
                     model = model, 
                     optstate = optstate, 
                     weights = weights})
    end

function get_dotp_dist_mem(split, example_num, t, keys, out_folder_name)
        local file_name_mem = out_folder_name.."/dotp_mem"..split..'_'..example_num.."_"..t..".txt"
        local dotp_mem = io.open(file_name_mem, "w")
        dotp_mem:write("dotproduct of mem cell contents and decoder matrix (vocab)\n")
        local decoder = getNodesByName("z").weight
        local memory = getNodesByName("memo"..t).output[example_num]
        local Hmem = getNodesByName("H").weight
        Hmem = Hmem:float()
        --Hmem = nn.View(1,opt.edim):forward(Hmem)
        memory = memory:float()
        local prod = nn.MM(false,true):forward({Hmem,memory})
        local prod_relu = nn.PReLU():forward(prod)
        prod_relu = prod_relu:cuda()
        local sample_data = nil
        if split == 'valid' then
            sample_data = tvdata
        elseif split == 'test' then
            sample_data = tedata
        end
        local prod_reluT = prod_relu:transpose(1,2)
        distances = {}
        local cosine = nn.CosineDistance()
        for i=1, opt.memslots do
                --local keiy = get_words(sample_data, keys[i],0)
                dotp_mem:write('\n')
                dotp_mem:write("\nContent of mem cell belonging to key  "..keys[i]..'\n')
                local vals = {}
                local x = prod_reluT[i]:float()
                for j=1, #sample_data.idict do
                     local y =  decoder[j]:float()
                     --vals[j] = cosine:forward({x, y})
                     local dist = cosine:forward({x, y})
                     vals[sample_data.idict[j]] = dist[1]
                     --vals[sample_data.idict[j]] = x * y
                end
                sortAttention_file(vals, dotp_mem)
        end
        dotp_mem:close()
end

function get_cosine_dist_mem(split, example_num,t, keys, out_folder_name)

        local file_name_mem = out_folder_name.."/cos_mem"..split..'_'..example_num.."_"..t..".txt"
        local cos_mem = io.open(file_name_mem, "w")
        cos_mem:write("Cosine similariry between  mem cell contents and entities (keys) \n")
        local decoder = getNodesByName("z").output[example_num]
        local memory = getNodesByName("memo"..t).output[example_num]
        local Hmem = getNodesByName("H").output[example_num]
        local key_embed = getNodesByName("keys").output[example_num]
        Hmem = Hmem:float()
        Hmem = nn.View(1,opt.edim):forward(Hmem)
        local sample_data = nil
        if split == 'valid' then
            sample_data = tvdata
        elseif split == 'test' then
            sample_data = tedata
        end
        distances = {}
        local cosine = nn.CosineDistance():float()
        for i=1, opt.memslots do
                --local keiy = get_words(sample_data, keys[i],0)
                cos_mem:write('\n')
                cos_mem:write("mem cell belongs to key "..keys[i]..'\n')
                local vals = {}
                local x = memory[i]:float()
                for j=1, opt.memslots do
                     --print("key embed is")
                     --print(key_embed)
                     local y = key_embed[1][j]:float()
                     vals[keys[j]] = cosine:forward({x, y})[1]
                     --print("val is fir i and j"..i..' '..j)
                     --print('Word '..keys[j]..' val '..vals[keys[j]])
                     --sortAttention(vals)
                end

                sortAttention_file(vals, cos_mem)
        end
        cos_mem:close()
end

local function get_time(story, question, data)
   -- story is 32 by 70 by 10
   -- question is 32 by 10
   -- get all the entities in the question
   story = torch.totable(story)
   question = torch.totable(question)
   local default = false
   local debug = false
   local timesteps = {}
   for i=1, opt.batchsize do
      local time = opt.T
      line = question[i]
      if line[4] == data.dict['look']  then
         local ent = line[3]
         local st = story[i]
         local reentry = false
         local orig_loc = ''
         for j = 1, opt.T do
            local stline = st[j]
            if stline[1] == ent and stline[2] == data.dict['entered'] then
               if reentry == false then
                   reentry = true
                   orig_loc = stline[4]
               else
                   if stline[4] == orig_loc then
                       time = opt.T
                   end
               end
            end
            if stline[1] == ent and stline[2] == data.dict['exited'] then
               time = j
             end
         end
     end
     if line[4] == data.dict['think'] then
         local ent1 = line[3]
         local ent2 = line[6]
         local st = story[i]
         local reentry = false
         local orig_loc = ''
         local time1 = opt.T
         local time2 =  opt.T
         for j = 1, opt.T do
            local stline = st[j]
            if stline[1] == ent1 and stline[2] == data.dict['entered'] then
               if reentry == false then
                   reentry = true
                   time1 = opt.T
                   orig_loc = stline[4]
               else
                   if stline[4] == orig_loc then
                       time1 = opt.T
                   end
               end
            end
            if stline[1] == ent1 and stline[2] == data.dict['exited'] then
               time1 = j
             end
         end
         st = story[i]
         reentry = false
         orig_loc = ''
         for j = 1, opt.T do
            local stline = st[j]
            if stline[1] == ent2 and stline[2] == data.dict['entered'] then
               if reentry == false then
                   reentry = true
                   time2 = opt.T
                   orig_loc = stline[4]
               else
                   if stline[4] == orig_loc then
                       time2 = opt.T
                   end
               end
            end
            if stline[1] == ent2 and stline[2] == data.dict['exited'] then
               time2 = j
             end
         end
         time = time2
         if time1 <= time then
             time = time1
         end
     end

     timesteps[i] = time     
   end
   return timesteps
end          

local function new_output_module(question, answer, story, facts, graph, data, time_file)
    -- story is 32 by 70 by 10
   local err, cost, missed, pred1, key_names = model:fprop(question, answer, story, graph)
   -- M is 32 by 8 by 100 Transpose 
   local sample_time =  get_time(story, question, data) 
   local M = torch.FloatTensor(opt.batchsize,opt.memslots,opt.edim)
   for i=1, #sample_time do
      local time = sample_time[i]
      time_file:write(time..'\n')
      local M_tmp = getNodesByName("memo"..time).output
      M[i] = M_tmp[i]:float()
   end
   
  local vocabsize = opt.nwords
  if opt.tied == 0 then
   -- add extra words to the vocabulary representing keys
      vocabsize = vocabsize + opt.memslots
  end
   local hid0 = getNodesByName("q").output
   local keys = getNodesByName("out_keys").output   
   --local M = getNodesByName("memo"..time).output
   hid0 = hid0:float()
   keys = keys:float()
   --M = M:float()
   --M = M_final
   local hid3dim = nn.View(opt.batchsize, 1, opt.edim):forward(hid0)
   local MM1aout = nn.MM(false, true)
   local MM2aout = nn.MM(false, true)
   local Aout = MM1aout:forward({hid3dim, M})
   local A2out = MM2aout:forward({hid3dim, keys})
   local Aout2dim = nn.View(opt.batchsize, -1):forward(Aout)
   local A2out2dim = nn.View(opt.batchsize,-1):forward(A2out)
   local P = nn.SoftMax():forward(Aout2dim)
   local P2 = nn.SoftMax():forward(A2out2dim)   
   local probs3dim = nn.View(1, -1):setNumInputDims(1):forward(P)--:annotate{name = "P2"}
   local probs23dim = nn.View(1, -1):setNumInputDims(1):forward(P2)--:annotate{name = "P2"}
   local MM1bout = nn.MM(false, false)
   local MM2bout = nn.MM(false, false)
   local Bout = nn.View(opt.batchsize, -1):forward(MM1bout:forward({probs3dim, M}))
   local B2out = nn.View(opt.batchsize, -1):forward(MM2bout:forward({probs23dim, keys}))
   local Hweights = getNodesByName("H").weight
   local C = nn.Linear(opt.edim, opt.edim, false)
   C.weight = Hweights:float()
   C = C:forward(Bout)
   local H2weights = getNodesByName("H2").weight
   local C2 = nn.Linear(opt.edim, opt.edim, false)
   C2.weight = H2weights:float()
   C2 = C2:forward(B2out)
   local D = nn.CAddTable():forward({hid0, C})
   local D2 = nn.CAddTable():forward({D, C2})
   local prelu_weights = getNodesByName("prelu").weight 
   local hid1 = nn.PReLU(opt.edim)
   hid1.weight = prelu_weights:float()   
   hid1 = hid1:forward(D2)
   local zweights = getNodesByName("z").weight
   local z = nn.Linear(opt.edim, vocabsize, false)
   z.weight = zweights:float()
   z = z:forward(hid1)
   local pred = nn.LogSoftMax():forward(nn.Narrow(2, 1, opt.nwords):forward(z))  
   local _, final_pred = pred:max(2)
   return err, cost, missed, pred1, key_names, final_pred
end


function evaluate_timetravel(split, out_folder_name)
    print("Evaluating time travel for "..split)
    local time_predictions_file, time_answers_file, time_questions_file, time_mypredictions_file, time_file = '', '','','',''
    if split == 'valid' then --unused
        N = val_range:size(1)
        indx = val_range
        data = tvdata

        time_predictions_file = io.open(out_folder_name.."/time_predictions.txt", "w")
        time_answers_file = io.open(out_folder_name.."/time_answers.txt","w")
        time_questions_file = io.open(out_folder_name.."/time_questions.txt","w")
        time_mypredictions_file = io.open(out_folder_name.."/time_mypredictions.txt", "w")
        time_file = io.open(out_folder_name.."/time.txt", "w")
    elseif split == 'test' then
        N = tedata.questions:size(1)
        indx = torch.range(1, N)
        data = tedata

        time_predictions_file = io.open(out_folder_name.."/time_predictions_test.txt", "w")
        time_answers_file = io.open(out_folder_name.."/time_answers_test.txt","w")
        time_questions_file = io.open(out_folder_name.."/time_questions_test.txt","w")
        time_mypredictions_file = io.open(out_folder_name.."/time_mypredictions_test.txt", "w")
        time_file = io.open(out_folder_name.."/time_test.txt", "w")
    end
    local loss = torch.Tensor(N)
    for k = 1, math.floor(N/opt.batchsize) do
        xlua.progress(k, math.floor(N/opt.batchsize))
        local batch = indx:index(1, torch.range(1 + (k-1)*opt.batchsize, k*opt.batchsize):long())
        local question, answer, story, facts, graph = data:getBatch(batch)
--        local err, cost, missed, pred, keys = model:fprop(question, answer, story, graph)
        local err, cost, missed, pred, keys, mypred = new_output_module(question, answer, story, facts, graph, data, time_file)
        key_names = keys
        for p = 1, opt.batchsize do
            a = answer[p]
            time_answers_file:write(get_words(data,a,0)..'\n')
            time_predictions_file:write(get_words(data,pred[p], 1)[1]..'\n')
            time_mypredictions_file:write(get_words(data,mypred[p], 1)[1]..'\n')
            q = get_words(data, question[p], 10)
            qs = '' 
            for i=1, #q do
                 qs = qs..q[i]..' '
            end
            time_questions_file:write(qs..'\n')
        end
    end
    time_predictions_file:close()
    time_answers_file:close()
    time_questions_file:close()
    time_mypredictions_file:close()
    time_file:close()
end

function test_timetravel(num, split)
    local batch = torch.FloatTensor(32):fill(num)
    local b = 0

    batch:apply(function()
        b = b + 1
        return b
    end)
    local num_sents = 6
    local sample_data = nil
    if split == 'valid' then
	    sample_data = tvdata
    elseif split == 'test' then
	    sample_data = tedata
    end

    local question, answer, story, facts, graph = sample_data:getBatch(batch)
    -- story is 32 by 70 by 10
    -- Lets take first 6 sentence of the story
    time_file = io.open("dummy.txt","w")
    local sents = {}
    local err, cost, missed, pred, key_names = model:fprop(question, answer, story, graph, timestep)
    local err, cost, missed, pred, keys, mypred = new_output_module(question, answer, story, facts, graph, data, time_file)
    for p = 1, opt.batchsize do
        print("pred is "..(get_words(sample_data,pred[p], 1)[1]..'\n'))
        print("mypred is "..(get_words(sample_data,mypred[p], 1)[1]..'\n'))
    end
end

evaluate_timetravel('valid', out_folder_name)
evaluate_timetravel('test', out_folder_name)

ret = run_example_and_plot('valid',1, out_folder_name)
get_dotp_dist_mem('valid', 1, 70, ret, out_folder_name)
get_cosine_dist_mem('valid', 1, 70, ret, out_folder_name)


ret = run_example_and_plot('valid',2, out_folder_name)
get_dotp_dist_mem('valid', 2, 70, ret, out_folder_name)
get_cosine_dist_mem('valid', 2, 70, ret, out_folder_name)
run_example_and_plot('valid',3, out_folder_name)
run_example_and_plot('valid',4, out_folder_name)
run_example_and_plot('valid',5, out_folder_name)
run_example_and_plot('valid',6, out_folder_name)


ret = run_example_and_plot('test',1, out_folder_name)
get_dotp_dist_mem('test', 1, 70, ret, out_folder_name)
get_cosine_dist_mem('test', 1,70, ret, out_folder_name)

ret = run_example_and_plot('test',2, out_folder_name)
get_dotp_dist_mem('test', 2, 70, ret, out_folder_name)
get_cosine_dist_mem('test', 2, 70, ret, out_folder_name)
run_example_and_plot('test',3, out_folder_name)
run_example_and_plot('test',4, out_folder_name)
run_example_and_plot('test',5, out_folder_name)
run_example_and_plot('test',6, out_folder_name)

if opt.load_pretrained_model ==0 then
    save_model()
end

