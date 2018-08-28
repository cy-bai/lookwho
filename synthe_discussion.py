from library_functions import *
np.set_printoptions(threshold=np.nan)

t_granularity = 10
NPLAYERS = 8
target_neighbors = [0] * (NPLAYERS + 1)
target_neighbors[0] = list(range(8))
for i in range(1, 9):
    target_neighbors[i] = [0, i, (i - 1 + 8) % 8, (i + 1) % 8]
print(target_neighbors)

def checkSyn(frame_order, frame_clip_st, lbls):
    totals=[]
    cnts= []
    for l in range(0, L - t_granularity, t_granularity):
        t = frame_order[l]
        next_t = frame_order[l + t_granularity]
        next_l = l + t_granularity
        lbl = lbls[:, t]
        next_lbl = lbls[:, next_t]

        if frame_clip_st[next_l] != 1:
            # print('aha')
            # print(lbl)
            # print(next_lbl)
            total = 0
            cnt = 0
            for ply in range(lbl.shape[0]):
                if ply!=-1 and next_lbl[ply]!=-1:
                    total+=1
                    if next_lbl[ply] in target_neighbors[lbl[ply]]:
                        cnt+=1
            assert cnt>0
            cnts.append(cnt)
            totals.append(total)
    print(np.sum(cnts)/np.sum(totals))

# for method in [0,4]:
   # load feat,label,clip_start_tag of introduction and test clips (two 5sec clips)
method = 4
MAX_L_PER_CLIP = 18 # 6sec
with open('feat/anno_intro_detc_fail_0/{}_game_frm_feat_dict.json'.format(feat_dict[method])) as f:
    game_intro_feat_dict = json.load(f)

game_time_order = {}
for game, intro_data in game_intro_feat_dict.items():
    intro_data = np.array(intro_data)
    N,L,D = intro_data.shape
    lbls = intro_data[:,:,-2].astype(np.int_)
    show = lbls.reshape((N,-1,t_granularity))[:,:,0]
    whole_selected_msk = np.zeros(L//t_granularity)
    process_time= 0

    selected_orders = []
    clip_st = []
    unselected_time = np.where(whole_selected_msk==0)[0]
    is_new_clip = True
    cur_clip_start_id = 0
    while unselected_time.shape[0] > 0:
        if is_new_clip:
            # random choose a unselected time to start
            # print('new clip')
            cur_time = np.random.choice(unselected_time)
            clip_st.append(1)
            cur_clip_start_id = len(clip_st)
        # else cur_time is selected by last time from the below "if"
        # mark cur_time, append to order list
        whole_selected_msk[cur_time] = 1
        selected_orders.append(cur_time)

        this_clip_tried_msk = whole_selected_msk.copy()
        # get players with label at cur_time
        plys_with_lbl = np.where(lbls[:,cur_time*t_granularity]>=0)[0]
        # random choose a player
        selected_ply = np.random.choice(plys_with_lbl)
        selected_lbl = lbls[selected_ply, cur_time*t_granularity]
        # print('### left {}, clip at {}, ply {},label {}'.format(unselected_time.shape[0], cur_time, selected_ply, selected_lbl))

        cur_clip_untried_times = np.where(this_clip_tried_msk==0)[0]
        while cur_clip_untried_times.shape[0] > 0:
            # print('  untried #', cur_clip_untried_times.shape)
            # randomly choose a time
            selected_time = np.random.choice(cur_clip_untried_times)
            # print(selected_time, selected_ply, lbls[selected_ply, selected_time * t_granularity])
            this_clip_tried_msk[selected_time] = 1
            if lbls[selected_ply, selected_time * t_granularity] in target_neighbors[selected_lbl]:
                # print(' select', selected_time, 'lbl', lbls[selected_ply, selected_time * t_granularity])
                break
            # else: mark this as tried, re select
            cur_clip_untried_times = np.where(this_clip_tried_msk == 0)[0]
        if cur_clip_untried_times.shape[0] == 0 or len(clip_st) - cur_clip_start_id + 1 >= MAX_L_PER_CLIP:
            # this clip has to end or clip length excceeds max length, restart new
            is_new_clip = True
        else:
            # continue this clip
            cur_time = selected_time
            clip_st.append(0)
            is_new_clip = False

        unselected_time = np.where(whole_selected_msk==0)[0]

    selected_orders = np.array(selected_orders)
    clip_st = np.array(clip_st)
    assert selected_orders.shape[0] == whole_selected_msk.shape[0]
    assert clip_st.shape[0] == whole_selected_msk.shape[0]
    assert np.max(selected_orders) == whole_selected_msk.shape[0]-1
    assert np.min(selected_orders) == 0
    print(game,'clip st', np.nonzero(clip_st)[0])
    frame_order = np.concatenate([range(start_id*t_granularity, (start_id+1)*t_granularity) for start_id in selected_orders], axis=0)
    frame_clip_st = np.zeros(L)
    frame_clip_st[np.nonzero(clip_st)[0]*t_granularity]=1
    # print('aha')
    # print(frame_order.shape, np.min(frame_order), np.max(frame_order))
    # print(game_time_order[game].shape)

    checkSyn(frame_order, frame_clip_st, lbls)
    game_time_order[game] = np.vstack((frame_order, frame_clip_st)).tolist()

with open('feat/anno_intro_detc_fail_0/intro_synthe1.json','w') as f:
    json.dump(game_time_order,f)
   # unsel_tids = np.where(selected_msk==0)[0]
   # tmp = np.random.randint(unsel_tids.shape[0])
   # time_id = unsel_tids[tmp]
   # print(unsel_tids)
   # print('sel', tmp, time_id)

