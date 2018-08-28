from library_functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--game_id', required=True, type=int, help='Game name')
parser.add_argument('--split_id', required=True, type=int, help='Number of splits')
parser.add_argument('--epochs', default=49, type=int, help='Number of epochs across the whole dataset')
parser.add_argument('--time_granularity', default=10, type=int, help='Number of frames used for training and testing')
parser.add_argument('--layer', default=3, type=int, help='Number of layers in the model')

args = parser.parse_args()

#### MAIN #####
# print command line arguments
# args: gameid [0-4], splitid [0-9]
root_dir = 'data_%dfrm/' % args.time_granularity
game_id = args.game_id
nsplits = args.split_id
print('gameid',game_id)

#nplayers = [6,7,7,8,7]
#feat_dim = [2,7,5,10,8]
#games = ['003SB','002ISR','004UMD','005NTU','006AZ']
#feat_dict = {0:'m0_gaze1',1:'m1_gaze1',2:'m2_gaze1',3:'m1_gaze2',4:'m2_gaze2'}
LR=1e-3
WEIGHT_DECAY=1e-5
STOP_L=0.5
NEPOCH = int(args.epochs)
NITER = int(args.layer)
time_granularity=int(args.time_granularity)

fold_tra_accs = np.zeros((nsplits, NEPOCH, NITER))
fold_tes_accs = np.zeros((nsplits, NEPOCH, NITER))
fold_tra_loss = np.zeros((nsplits, NEPOCH, NITER))
fold_tes_loss = np.zeros((nsplits, NEPOCH, NITER))
for split_id in range(nsplits):
    fold_tra_accs[split_id, :, :] = np.loadtxt('{}/eval/{}_{}_tra_acc.txt'.format(root_dir, game_id, split_id))
    fold_tes_accs[split_id, :, :] = np.loadtxt('{}/eval/{}_{}_tes_acc.txt'.format(root_dir, game_id, split_id))
    fold_tra_loss[split_id, :, :] = np.loadtxt('{}/eval/{}_{}_tra_loss.txt'.format(root_dir, game_id, split_id))
    fold_tes_loss[split_id, :, :] = np.loadtxt('{}/eval/{}_{}_tes_loss.txt'.format(root_dir, game_id, split_id))

game = GAMES[game_id]
# plot
plot_epochs = NEPOCH
tra_acc = np.mean(fold_tra_accs, axis=0)[:plot_epochs]
tes_acc = np.mean(fold_tes_accs, axis=0)[:plot_epochs]
tra_loss = np.mean(fold_tra_loss, axis=0)[:plot_epochs]
tes_loss = np.mean(fold_tes_loss, axis=0)[:plot_epochs]
print(tes_acc)
plt.figure(figsize = (8,4))
plt.grid(True)
plt.title(game)
plt.xlabel('epoch')
plt.plot(range(plot_epochs), tra_acc[:, 0], 'g-x', tra_acc[:, 1], 'b-x', tra_acc[:, 2], 'r-x',
         tes_acc[:, 0], 'g-', tes_acc[:, 1], 'b-', tes_acc[:, 2], 'r-')
plt.legend(['tra acc lay1', 'tra acc lay2', 'tra acc lay3',
            'tes acc lay1', 'tes acc lay2', 'tes acc lay3'])
plt.savefig('{}/fig/{}_acc.png'.format(root_dir, game))


plt.figure(figsize=(8,4))
plt.grid(True)
plt.title(game)
plt.xlabel('epoch')
plt.plot(range(plot_epochs), tra_loss[:, 0], 'g-x', tra_loss[:, 1], 'b-x', tra_loss[:, 2], 'r-x',
         tes_loss[:, 0], 'g-', tes_loss[:, 1], 'b-', tes_loss[:, 2], 'r-')
plt.legend(['tra loss lay1', 'tra loss lay2', 'tra loss lay3',
            'tes loss lay1', 'tes loss lay2', 'tes loss lay3'])
plt.savefig('{}/fig/{}_loss.png'.format(root_dir, game))
