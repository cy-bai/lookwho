import os,sys,json, sklearn
import pandas as pd
import numpy as np
from glob import glob
from math import atan2
import sklearn.ensemble, sklearn.metrics,sklearn.svm
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy.stats import mode
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import gpustat

LR=1e-2
WEIGHT_DECAY=1e-5
STOP_L=0.5
GAMES = ['003SB','002ISR','004UMD','005NTU','006AZ']
NPLAYERS = [6,7,7,8,7]
FEAT_DIM = [2,7,5,10,8]
METHOD=4
feat_dict = {0:'m0_gaze1',1:'m1_gaze1',2:'m2_gaze1',3:'m1_gaze2',4:'m2_gaze2'}

# split test clips, leave 1 sec out
def genTestCVIdx(game_data, leave_m=5):
    # N players, L time length, D feature dim
    # currently game data only has two 5sec clips for each player
    N,L,D = game_data.shape
    D -= 2
    fps=30
    tra_tes_idx = []
    for i in range(0,5*fps,leave_m*fps):
        test_idx = np.arange(i,i+leave_m*fps)
        train_idx = np.hstack((np.arange(0,i),np.arange(5*fps,10*fps)))
        tra_tes_idx.append((train_idx, test_idx))
        assert np.intersect1d(train_idx, test_idx).shape[0] == 0

    for i in range(5*fps,10*fps, leave_m*fps):
        test_idx = np.arange(i,i+leave_m*fps)
        train_idx = np.hstack((np.arange(5*fps,i),np.arange(0,5*fps)))
        tra_tes_idx.append((train_idx, test_idx))
        assert np.intersect1d(train_idx, test_idx).shape[0] == 0
    return tra_tes_idx

# splot test clips, time_granularity controls how many frames to test, gap controls #seconds of gap
def TgapTestCVIdx(game_data, gap=0, time_granularity = 10):
    fps = 30
    #time_granularity = 10
    step = time_granularity
    tra_tes_idx = []
    if gap<14:
        for i in range(0+step, 5*fps, step):
            tra_idx_ed = i - gap*fps
            if tra_idx_ed <= 0:
                continue
            tes_idx = np.arange(i,i+time_granularity)
            tra_idx = np.hstack((np.arange(0,tra_idx_ed), np.arange(5*fps,10*fps)))
            tra_tes_idx.append((tra_idx, tes_idx))
            assert np.intersect1d(tra_idx, tes_idx).shape[0] == 0
        for i in range(5*fps+step, 10*fps, step):
            tra_idx_ed = i - gap*fps
            if tra_idx_ed <= 5*fps:
                continue
            tes_idx = np.arange(i,i+time_granularity)
            tra_idx = np.hstack((np.arange(5*fps,tra_idx_ed), np.arange(0, 5*fps)))
            tra_tes_idx.append((tra_idx, tes_idx))
            assert np.intersect1d(tra_idx, tes_idx).shape[0] == 0

        # start of a clip
        tra_tes_idx.append((np.arange(0, 5 * fps), np.arange(5*fps, 5*fps + time_granularity)))
        tra_tes_idx.append((np.arange(5 * fps, 10 * fps), np.arange(0, time_granularity)))

    # else:
    #     # train on one 5sec clip, test on any sec of another clip
    #     for i in range(0, 5*fps, step):
    #         tra_tes_idx.append((np.arange(5*fps, 10*fps), np.arange(i,i+time_granularity)))
    #     for i in range(5*fps,10*fps,step):
    return tra_tes_idx

# compute ACC for each 10 frm
def computeNFrmACC(y_score, y_true, N=10):
    Nfrm_score = np.mean(y_score.reshape((-1,N, y_score.shape[1])), axis=1)
    Nfrm_true = mode(y_true.reshape((-1,N)), axis=1)[0].ravel()
    Nfrm_pred = np.argmax(Nfrm_score, axis=1).ravel()
    valid_msk = Nfrm_true >=0
    if np.sum(valid_msk)==0:
        # label for this player, this fold is unknown!!!!!
        # return a tag so that this fold doesn't count
        return -1
#     print(y_true[valid_msk].shape)
    acc = sklearn.metrics.accuracy_score(y_pred=Nfrm_pred[valid_msk], y_true=Nfrm_true[valid_msk])
#     print(Nfrm_pred[valid_msk], Nfrm_true[valid_msk])
    return acc

# train player-based single RF as layer 0's output
def warmStart(X,y, tes_X):
    X,y = X.numpy().copy(),y.numpy().copy()
    tes_X = tes_X.numpy().copy()
    N,L,D = X.shape
    estimators=[]
    tes_y_init = []
    y_init = []
    for i in range(N):
        msk = y[i,:]>=0
        X_ply,y_ply = X[i,msk,:], y[i,msk]
        estimator, feat_im = OVOTrain(X_ply,y_ply)
        y_prob = estimator.predict_proba(X[i,:])
        # labels are (0,1,..,i-1,i+1,N)
        y_init.append(np.insert(y_prob, i+1, np.zeros(y_prob.shape[0]), axis = 1)[np.newaxis,:])
        tes_y_prob = estimator.predict_proba(tes_X[i,:])
        tes_y_init.append(np.insert(tes_y_prob, i+1, np.zeros(tes_y_prob.shape[0]), axis = 1)[np.newaxis,:])
#         print(np.nonzero(np.max(tes_y_prob, axis=1) > 0.9))
    y_init = np.concatenate(y_init,axis=0)
    tes_y_init = np.concatenate(tes_y_init,axis=0)
    return torch.tensor(y_init,dtype=torch.float), torch.tensor(tes_y_init, dtype=torch.float)

# one vs one training
def OVOTrain(X,y):
    estimator = sklearn.ensemble.RandomForestClassifier(n_estimators=30, class_weight='balanced', random_state=0)
#     estimator = sklearn.svm.SVC(class_weight='balanced',decision_function_shape='ovo',kernel='linear',random_state=0, probability=True)
#     estimator = sklearn.linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
    estimator.fit(X,y)
    feat_importance=None
#     feat_importance = estimator.feature_importances_
    return estimator, feat_importance

# data normalization, not used
def normalize(X):
    return (X-np.min(X,axis=0)) / (np.max(X,axis=0) - np.min(X,axis=0))

# get class weights, used for loss functions
# y is N*L, where N is Nplayer, L is time length
# w is N*(N+1), w(i,j) denotes weights if player (i+1) lookat player j, w(:,0) is lookat-laptop
# w(i,i+1) is 1, since one cannot look at himself, it is the largest penalty
def getWgt(y):
    N,L = y.size()
    player_weights = torch.zeros((N,N+1))
    for i in range(N):
        y_np = y[i,:].numpy().copy()
        y_np = y_np[y_np>=0]
        cnt = Counter(y_np.astype(np.int_))
#         print('player',i+1,cnt)
        w = torch.zeros(N+1)
        for k,v in cnt.items():
            w[k] = y_np.shape[0] / v
        player_weights[i] = w
        player_weights[i] = player_weights[i] / torch.sum(player_weights[i])
        player_weights[i,i+1]=100
#     player_weights = torch.tensor(player_weights / torch.sum(player_weights), dtype=torch.float)
    return player_weights

def splitToCLips(X,y,layer_0_out,np_clip_st):
    start_idxs = np.where(np_clip_st>0)[0]
    assert start_idxs[0] == 0
    Xs,ys,lay0_outs = [],[],[]
    for i in range(start_idxs.shape[0]-1):
        Xs.append(X[:,start_idxs[i]:start_idxs[i+1],:])
        ys.append(y[:,start_idxs[i]:start_idxs[i+1]])
        lay0_outs.append(layer_0_out[:,start_idxs[i]:start_idxs[i+1],:])
    Xs.append(X[:,start_idxs[-1]:,:])
    ys.append(y[:,start_idxs[-1]:])
    lay0_outs.append(layer_0_out[:,start_idxs[-1]:,:])
    return Xs,ys, lay0_outs

class NN(nn.Module):
    def __init__(self, rawfeat_dim, tag_dim):
        super(NN, self).__init__()
        self.tag_dim = tag_dim
        self.i2o = nn.Linear(rawfeat_dim + tag_dim*3, tag_dim)
#         self.relu = nn.ReLU()
    # feat: raw feature, recur: last layer output, collect: avg(others), temp: last time output
    def forward(self, feat, recur, collect, temp):
#         print(feat.size(), collect.size(), temp.size(), recur.size())
        concat = torch.cat([feat, collect, temp, recur]).unsqueeze(0)
        output = self.i2o(concat)
        return output

class RNN(nn.Module):
    def __init__(self, rawfeat_dim, tag_dim):
        super(RNN, self).__init__()
        self.rnn = nn.RNNCell(rawfeat_dim + 2 * tag_dim, tag_dim)
    def forward(self, rawfeat, recur, collect, temp):
        input = torch.cat([rawfeat, collect, temp]).unsqueeze(0)
        output = self.rnn(input, recur.unsqueeze(0))
        return output

def NNInit(N,D,w):
    # N players, D raw feature dim
    tag_sz = N+1
    player_nns, params, criterions = [], [], []
    for i in range(N):
        # player_nns.append(NN(D, tag_sz).cuda())
        player_nns.append(RNN(D,tag_sz).cuda())
        params = params + list(player_nns[i].parameters())
        # player-dependent weights
        criterions.append(nn.CrossEntropyLoss(weight=w[i]))
    opt = optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    return player_nns, criterions, opt

def train(Xs, ys, w, layer_0_outs, tes_X, tes_y, tes_lay0_out,
          root_dir, game_id, split_id, NEPOCH, NITER):
    torch.manual_seed(1)
    # N players, L time length, D feature dim
    N,D = Xs[0].size()[0], Xs[0].size()[2]
    # num of clips
    C = len(Xs)
    # initialization player nns, criterions, optimizer
    w = w.cuda()
    player_nns, criterions, opt = NNInit(N,D,w)
    # training
    epoch_loss_lst = []
    for epoch in range(NEPOCH):
        running_loss = 0.
        for clip in range(C):
            X,y = Xs[clip].cuda(), ys[clip].cuda()
            # init last layer's out
            last_lay_out = layer_0_outs[clip].cuda()
            # init t0 out as uniform distribution
            time_0_out = torch.ones(1+N).float()/(1+N)
            time_0_out = time_0_out.cuda()
            # clear grad
            opt.zero_grad()

            loss_lst = []
            L_clip = y.size()[1]
            # each layer
            # print('forward clip:', clip, L_clip, 'frms')
            for iter in range(NITER):
                this_lay_out = torch.zeros(N, L_clip, N+1).cuda()
                collect_last_lay_out = torch.cat([constructCollect(last_lay_out, ply) for ply in range(N)],dim=0)
                for ply in range(N):
                    for t in range(L_clip):
                        last_t_out = time_0_out if t==0 else this_lay_out[ply,t-1,:]
                        out = player_nns[ply](X[ply,t,:], last_lay_out[ply,t,:],
                                              collect_last_lay_out[ply][t,:], last_t_out)
                        this_lay_out[ply,t,:] = F.softmax(out, dim=1)[0]
                    # only compute loss for training sets with label
                    sure_msk = y[ply,:]>=0
                    if sure_msk.long().sum().item() > 0:
                        loss_lst.append(criterions[ply](this_lay_out[ply,sure_msk,:], y[ply,sure_msk]))
#                     else:
#                         print('ply',ply,'clip',clip,'no label!')
                # update layer's output
                last_lay_out = this_lay_out
            # back prop
            clip_loss = sum(loss_lst)
            # print('before')
            # for ply in range(N):
            #     for name, param in player_nns[ply].named_parameters():
            #         print(name, param.data)
            clip_loss.backward()
            opt.step()
            # print('after')
            # for ply in range(N):
            #     for name, param in player_nns[ply].named_parameters():
            #         print(name, param.data)

            # print(clip_loss.item())

            running_loss += clip_loss.item()/NITER
        # print('epoch ',epoch, running_loss)
        # save model for this epoch
        for ply in range(N):
            saveModel(root_dir, game_id, ply, split_id, player_nns[ply], opt, epoch, NITER)

        #normalize by #clips
        running_loss /= C
        # early stop
        # if running_loss < STOP_L:
        #     break
        # print('epoch',epoch,'loss', running_loss)
        # tra_acc = []
        # for clip in range(C):
        #     tra_acc.append(evalNNCC(player_nns, Xs[clip].cuda(), ys[clip].cuda(), layer_0_outs[clip].cuda()))
        # print('tra acc', np.mean(np.array(tra_acc),axis=0))
        # print('test acc',evalNNCC(player_nns, tes_X,tes_y,tes_lay0_out))
        epoch_loss_lst.append(running_loss)
        np.savetxt('{}/tra/epoch_losses_{}_{}.txt'.format(root_dir, game_id, split_id),np.array(epoch_loss_lst), fmt='%.4f')
    return player_nns, epoch_loss_lst


def saveModel(root_dir, game_id, ply, split_id, model, opt, epoch, NITER):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict(),
        'n_layers': NITER
    }
    fnm = '{}/model/{}_{}_{}_{}.pt'.format(root_dir, game_id, ply, split_id, epoch)
    torch.save(state, fnm)

# lastlay_outs: N*L*(N+1)
# ply's collective feature (construct from other players)
def constructCollect(lastlay_outs, ply):
    N,L,D = lastlay_outs.size()
    # option 1, using all player's outputs
#     other_outputs = player_outputs
    # option 2, without cur player
    other_outputs = torch.cat((lastlay_outs[:ply], lastlay_outs[ply+1:]), dim=0)
    # method1, concate
#     collec_x = np.swapaxes(other_outputs, 0,1).reshape((L,-1))
    # method2, avg
    collec_x = torch.mean(other_outputs, dim=0)
    return collec_x.unsqueeze(0)

def eval_layerForward(player_nns, last_lay_out, clip_X, clip_y):
    N, L_clip, D = last_lay_out.size()
    this_lay_out = torch.zeros_like(last_lay_out).cuda()
    collect_last_lay_out = torch.cat([constructCollect(last_lay_out, ply) for ply in range(N)],0)
    time_0_out = torch.ones(1+N)/(1+N)
    time_0_out = time_0_out.cuda()
    for ply in range(N):
        for t in range(L_clip):
            last_t_out = time_0_out if t ==0 else this_lay_out[ply,t-1,:]
            out = player_nns[ply](clip_X[ply,t,:], last_lay_out[ply,t,:],
                                  collect_last_lay_out[ply][t,:], last_t_out).detach()
            this_lay_out[ply,t,:] = F.softmax(out, dim=1)[0]
    return this_lay_out

def evalNNCC(player_nns, criterions, X,y,lay_0_out, NITER):
    # print('eval', X.size(), y.size())
    X,y = X.cuda(),y.cuda()
    N,L,D=X.size()
    last_lay_out = lay_0_out.cuda()
    for ply in range(N):
        player_nns[ply].eval()
    game_lay_acc = []
    game_lay_loss = []
    for iter in range(NITER):
        ply_lay_acc = []
        ply_lay_loss = []
        last_lay_out = eval_layerForward(player_nns, last_lay_out, X, y)
        for ply in range(N):
            sure_msk = y[ply, :] >= 0
            if sure_msk.long().sum().item() > 0:
                ply_lay_loss.append(criterions[ply](last_lay_out[ply, sure_msk, :], y[ply, sure_msk]))
                ply_lay_acc.append(computeNFrmACC(last_lay_out[ply].cpu().numpy(), y[ply].cpu().numpy()))

        game_lay_acc.append(smartMean(ply_lay_acc))
        game_lay_loss.append(np.mean(np.array(ply_lay_loss)))
    return game_lay_loss, game_lay_acc

def evalTrain(player_nns, criterions, Xs, ys, lay0_out, NITER):
    torch.manual_seed(1)
    # N players, L time length, D feature dim
    # N, D = Xs[0].size()[0], Xs[0].size()[2]
    # num of clips
    C = len(Xs)
    game_lay_acc = []
    game_lay_loss = []
    running_loss = 0.
    for clip in range(C):
        clip_lay_loss, clip_lay_acc = evalNNCC(player_nns, criterions, Xs[clip], ys[clip], lay0_out[clip], NITER)
        game_lay_acc.append(clip_lay_acc)
        game_lay_loss.append(clip_lay_loss)
        
    game_lay_loss = np.array(game_lay_loss)
    game_lay_acc = np.array(game_lay_acc)
    running_loss = np.mean(game_lay_loss[:, -1])
    # print('epoch loss', running_loss)
    assert game_lay_loss.shape[0]==C
    return np.mean(game_lay_loss, axis=0), np.mean(game_lay_acc, axis=0)
    
def smartMean(acc_arr):
    acc_arr = np.array(acc_arr)
    acc_arr = acc_arr[acc_arr!=-1]
    assert acc_arr.shape[0]>0
    return np.mean(acc_arr)

def select_free_gpu():
   mem = []
   gpus = list(set([0,1,2,3]))
   for i in gpus:
       gpu_stats = gpustat.GPUStatCollection.new_query()
       mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
   return str(gpus[np.argmin(mem)])


def NNInitFromFile(root_dir, game_id, split_id, epoch, w):
    # N players, D raw feature dim
    player_nns, criterions = [], []
    N = NPLAYERS[game_id]
    D = FEAT_DIM[METHOD]
    tag_sz = N+1
    for i in range(N):
        fnm = '{}/model/{}_{}_{}_{}.pt'.format(root_dir, game_id, i, split_id, epoch)
        # print(fnm)
        # print(fnm)
        model = RNN(D, tag_sz)
        state = torch.load(fnm)
        model.load_state_dict(state['state_dict'])
        player_nns.append(model.cuda())
        # player-dependent weights
        criterions.append(nn.CrossEntropyLoss(weight=w[i]))
    return player_nns, criterions