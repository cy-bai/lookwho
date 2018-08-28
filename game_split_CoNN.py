from library_functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--game_id', required=True, type=int, help='Game name')
parser.add_argument('--split_id', required=True, type=int, help='Game name')
parser.add_argument('--epochs', default=49, type=int, help='Number of epochs across the whole dataset')
parser.add_argument('--time_granularity', default=10, type=int, help='Number of frames used for training and testing')
parser.add_argument('--layer', default=3, type=int, help='Number of layers in the model')
parser.add_argument('--synthe_name', default='none', type=str, help='name of synthetic intro order json')
args = parser.parse_args()

if args.game_id >= len(GAMES):
   print("GAME ID IS NOT VALID. EXITING")
   sys.exit(0)

gpu = select_free_gpu()
print("gpu", gpu)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

np.set_printoptions(threshold=np.nan)
torch.manual_seed(1)
np.random.seed(0)
# cuda_avail = torch.cuda.is_available()
#cuda_avail = True
# print('version',torch.__version__)
#print('cuda', torch.cuda.is_available())

root_dir = './data_%dfrm/'% args.time_granularity
game_id = args.game_id
game = GAMES[game_id]
split_id = args.split_id
synthe_name = args.synthe_name
print('gameid',game_id, 'splitid', split_id)

NEPOCH = int(args.epochs)
NITER = int(args.layer)
time_granularity=int(args.time_granularity)

##### MAIN #####

for method in [4]:
   # load feat,label,clip_start_tag of introduction and test clips (two 5sec clips)
    with open('feat/anno_intro_detc_fail_0/{}_game_frm_feat_dict.json'.format(feat_dict[method])) as f:
       game_intro_feat_dict = json.load(f)
    with open('feat/test_feat_with_gt_detec_fail/{}_game_frm_feat_dict.json'.format(feat_dict[method])) as f:
       game_dict = json.load(f)
       # data[i,j,:-2] is feature of player i, time j, data[i,j,-2] is label,
       # data[i,j,-1] > 0 iff this is start of a clip, data[i,:,-1] is equal for every i

    data = game_dict[game]
    intro_data = game_intro_feat_dict[game]
    #synthe order
    if synthe_name !='none':
        with open('feat/anno_intro_detc_fail_0/{}.json'.format(synthe_name)) as f:
            order_clipst = np.array(json.load(f)[game]).astype(np.int_)
        intro_data = np.array(intro_data)
        intro_data = intro_data[:,order_clipst[0],:]
        intro_data[:,:,-1]=order_clipst[1]

    # tra_tes_idxs = genTestCVIdx(np.array(data), 1)
    tra_tes_idxs = TgapTestCVIdx(np.array(data), 0, time_granularity)
    intro_data_tensor = torch.tensor(intro_data, dtype=torch.float)
    data_tensor = torch.tensor(data, dtype=torch.float)
    game_acc = []
    print('start training and testing of game-split:', game,split_id)
    tra_idx, tes_idx = tra_tes_idxs[split_id]
    # training X=[intro_X, tra_X], testing tes_X
    intro_X, intro_y, intro_clip_st = intro_data_tensor[:,:,:-2],intro_data_tensor[:,:,-2].long(), intro_data_tensor[:,:,-1].long()
    tra_X, tra_y, tra_clip_st = data_tensor[:,tra_idx,:-2], data_tensor[:,tra_idx,-2].long(), data_tensor[:,tra_idx,-1].long()
    tes_X, tes_y, tes_clip_st = data_tensor[:,tes_idx,:-2], data_tensor[:,tes_idx,-2].long(), data_tensor[:,tes_idx,-1].long()
    
    X, y, clip_st = torch.cat((intro_X, tra_X), 1), torch.cat((intro_y, tra_y),1), torch.cat((intro_clip_st, tra_clip_st),1)
    # print('data sizes', intro_X.size(), tra_X.size(), X.size(), intro_y.size(), tra_y.size(), y.size(), tes_X.size(), tes_y.size())
    # train player-based RFs and get predicted probs for all data
    #lay0_out(i,j,:) is player i's predicted probs at time j
    lay0_out, tes_lay0_out = warmStart(X,y,tes_X)
    # split data to clips, clip_st is the same for all players
    Xs,ys,layer_0_outs = splitToCLips(X,y,lay0_out,clip_st[0].numpy())
    # train
    player_nns, train_loss_lst = train(Xs, ys, getWgt(intro_y), layer_0_outs, tes_X, tes_y, tes_lay0_out,
                                       root_dir, game_id, split_id, NEPOCH, NITER)


# In[10]:


# # ply_lastlay_outs: N*L*(N+1)
# def constructCollect(ply_lastlay_outs, ply, ply_clip_st=None, intro_L=0):
#     ply_lastlay_outs = F.softmax(ply_lastlay_outs, dim=2)
#     N,L,D = ply_lastlay_outs.size()
#     # option 1, using all player's outputs
# #     other_outputs = player_outputs
#     # option 2, without cur player
#     other_outputs = torch.cat((ply_lastlay_outs[:ply], ply_lastlay_outs[ply+1:]), dim=0)
#     # method1, concate
# #     relational_x = np.swapaxes(other_outputs, 0,1).reshape((L,-1))
#     # method2, avg
#     relational_x = torch.mean(other_outputs, dim=0)
# #     if ply_clip_st is not None:
#         # add last time last layer's output
# #         last_t_last_l = consLastTimeOutput(ply_lastlay_outs[ply], ply_clip_st, N, intro_L)
# #         return torch.cat([relational_x, ply_lastlay_outs[ply], last_t_last_l],dim=1)
#     return relational_x


# In[11]:


# def consLastTimeOutput(ply_lastlay_out, clip_st, Nplayer, intro_L):
#     t0_out = torch.ones(Nplayer+1)/(Nplayer+1.)
#     assert clip_st[0]>0
#     last_time_out = torch.empty_like(ply_lastlay_out)
#     for i in range(last_time_output.size()[0]):
#         # for start fram of a clip, set prob as uniform
#         if clip_st[i]>0:
#             last_time_out[i,:] = t0_out
#         else:
#             last_time_out[i,:] = ply_lastlay_out[i-1,:]
#     return last_time_output


# In[12]:


# with open('../synced_feats/anno_intro_feat/game_ply_talk_prob.json') as f:
#     game_intro_talk_dict = json.load(f)
# with open('../synced_feats/test_feat/game_ply_talk_prob.json') as f:
#     game_talk_dict = json.load(f)
# def addOthersTalkProb(intro_prob, intro_data, tes_prob, tes_data):
#     N,L,D = intro_data.shape
#     tes_L = tes_data.shape[1]
# #     print(intro_prob.shape, L, tes_prob.shape,tes_L)
#     intro_talk_prob = np.zeros((N,L,N))
#     tes_talk_prob = np.zeros((N, tes_L, N))
#     for ply in range(N):
#         intro_talk_prob[ply,:,:] = intro_prob.T.copy()
#         # don't include self talk prob!
#         intro_talk_prob[ply,:,ply] = 0
#         tes_talk_prob[ply,:,:] = tes_prob.T.copy()
#         tes_talk_prob[ply,:,ply] = 0
#     intro_data = np.concatenate([intro_data[:,:,:-2], intro_talk_prob, intro_data[:,:,-2:]], axis=2)
#     tes_data = np.concatenate([tes_data[:,:,:-2], tes_talk_prob, tes_data[:,:,-2:]], axis=2)
#     return intro_data, tes_data

