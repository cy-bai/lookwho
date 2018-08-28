from library_functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--game_id', required=True, type=int, help='Game name')
parser.add_argument('--split_id', required=True, type=int, help='Game name')
parser.add_argument('--epochs', default=49, type=int, help='Number of epochs across the whole dataset')
parser.add_argument('--time_granularity', default=10, type=int, help='Number of frames used for training and testing')
parser.add_argument('--layer', default=3, type=int, help='Number of layers in the model')
args = parser.parse_args()

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

##### MAIN #####
# print command line arguments
# args: gameid [0-4], splitid [0-9]
root_dir = './data_%dfrm/' % args.time_granularity
game_id = args.game_id
game = GAMES[game_id]
split_id = int(args.split_id)
print('gameid',game_id, 'splitid', split_id)

LR=1e-3
WEIGHT_DECAY=1e-5
STOP_L=0.5
NEPOCH = int(args.epochs)
NITER = int(args.layer)
time_granularity=int(args.time_granularity)


for method in [METHOD]:
    # load feat,label,clip_start_tag of introduction and test clips (two 5sec clips)
    with open('feat/anno_intro_detc_fail_0/{}_game_frm_feat_dict.json'.format(feat_dict[method])) as f:
        game_intro_feat_dict = json.load(f)
    with open('feat/test_feat_with_gt_detec_fail/{}_game_frm_feat_dict.json'.format(feat_dict[method])) as f:
        game_dict = json.load(f)

    # for game in ['003SB','002ISR','004UMD','005NTU','006AZ']:
    #for game in [games[game_id]]:
    # data[i,j,:-2] is feature of player i, time j, data[i,j,-2] is label,
    # data[i,j,-1] > 0 iff this is start of a clip, data[i,:,-1] is equal for every i
    data = game_dict[game]
    intro_data = game_intro_feat_dict[game]

    # tra_tes_idxs = genTestCVIdx(np.array(data), 1)
    tra_tes_idxs = TgapTestCVIdx(np.array(data), 0, time_granularity)
    intro_data_tensor = torch.tensor(intro_data, dtype=torch.float)
    data_tensor = torch.tensor(data, dtype=torch.float)
    
    fold_tra_accs = np.zeros((NEPOCH, NITER))
    fold_tes_accs, fold_tra_loss, fold_tes_loss= fold_tra_accs.copy(), fold_tra_accs.copy(), fold_tra_accs.copy()
    print('start eval of game-split:', game_id)
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
    w = getWgt(intro_y).cuda()
    
    for epoch in range(NEPOCH):
       # print('epoch',epoch)
       player_nns, criterions = NNInitFromFile(root_dir, game_id, split_id, epoch, w)
       loss,acc = evalTrain(player_nns, criterions, Xs, ys, layer_0_outs, NITER)
       fold_tra_accs[epoch,:]=np.array(acc)
       fold_tra_loss[epoch,:]=np.array(loss)
       loss,acc = evalNNCC(player_nns, criterions, tes_X, tes_y, tes_lay0_out, NITER)
       fold_tes_accs[epoch,:]=np.array(acc)
       fold_tes_loss[epoch,:]=np.array(loss)
       # print(id,epoch,'traacc',fold_tra_accs[epoch])
       # print(id,epoch,'tesacc',fold_tes_accs[epoch])
       
       np.savetxt('{}/eval/{}_{}_tra_acc.txt'.format(root_dir, game_id, split_id), fold_tra_accs, fmt='%.4f')
       np.savetxt('{}/eval/{}_{}_tes_acc.txt'.format(root_dir, game_id, split_id), fold_tes_accs, fmt='%.4f')
       np.savetxt('{}/eval/{}_{}_tra_loss.txt'.format(root_dir, game_id, split_id), fold_tra_loss, fmt='%.4f')
       np.savetxt('{}/eval/{}_{}_tes_loss.txt'.format(root_dir, game_id, split_id), fold_tes_loss, fmt='%.4f')
