require('paths')
require('nngraph')
require('cunn')
require('optim')
paths.dofile('params.lua')
paths.dofile('utils.lua')
paths.dofile('layers/Normalization.lua')
torch.setdefaulttensortype('torch.FloatTensor')
g_make_deterministic(123)

function printNet(net)
    for i = 1, net:size(1) do
        print(string.format("%d: %s", i, net.modules[i]))
    end
end


--trdata = paths.dofile('data.lua')
--tedata = paths.dofile('data.lua')
--trdata:load(trfiles, opt.batchsize, opt.T)
--tedata:load(tefiles, opt.batchsize, trdata.memsize, trdata.dict, trdata.idict)
save_file = 'outputs-original-new/final.model'
--task=21-model=entnet-embed=icmul+bow-edim=100-initstd=0.1-sdt_decay=25-optim=adam-sdt=0.01-nhop=1-tied=1-T=70_init_num_0.model'
--..opt.modelFilename .. '_init_num_' .. opt.save_num .. '.model'
print("Save file is"..save_file)
obj = torch.load(save_file)
model = obj['model']
print("model loaded")
--graph.dot(model.network.fg, 'graph', 'mygraph')
print("Printing the network")
printNet(model.network)
print('\nmodel: ' .. paths.basename(opt.modelFilename))
print('#params = ' .. model.paramx:size(1))
