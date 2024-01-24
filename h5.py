import h5py
import numpy as np

h5f = h5py.File('dataset/16740.h5','r')
game_log = np.array(h5f['16740'][:])
print(game_log)
h5f.close()

np.savetxt('status.txt',game_log[0][59])
np.savetxt('moves.txt',game_log[1][0])