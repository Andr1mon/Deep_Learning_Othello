import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np
import os
import sys
import h5py
import json
from tqdm import tqdm
from datetime import datetime
import h5py
import copy

from utile import has_tile_to_flip, isBlackWinner, initialze_board, BOARD_SIZE
from networks_e2205046 import MLP, LSTMs


class SampleManager():
    def __init__(self,
                 game_name,
                 file_dir,
                 end_move,
                 len_moves,
                 isBlackPlayer):
        
        ''' each sample is a sequence of board states 
        from index (end_move - len_moves) to inedx end_move
        
        file_dir : directory of dataset
        game_name: name of file (game)
        end_move : the index of last recent move 
        len_moves: length of sequence
        isBlackPlayer: register the turn : True if it is a move of black player
        	(if black is the current player the board should be multiplay by -1)
        '''
        
        self.file_dir=file_dir
        self.game_name=game_name
        self.end_move=end_move
        self.len_moves=len_moves
        self.isBlackPlayer=isBlackPlayer
    
    def set_file_dir(file_dir):
        self.file_dir=file_dir
    def set_game_name(game_name):
        self.game_name=game_name
    def set_end_move(end_move):
        self.end_move=end_move
    def set_len_moves(len_moves):
        self.len_moves=len_moves


class CustomDataset(Dataset):
    def __init__(self,
                 dataset_conf,load_data_once4all=True):
        """
        Custom dataset class for Othello game.

        Parameters:
        - dataset_conf (dict): Configuration dictionary containing dataset parameters.
        - load_data_once4all (bool): Flag indicating whether to load all data at once.
        """
                 
        self.load_data_once4all=load_data_once4all
        
        self.starting_board_stat=initialze_board()
        
        # self.filelist : a list of all games for train/dev/test
        self.filelist=dataset_conf["filelist"]
        #len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
        self.len_samples=dataset_conf["len_samples"] 
        self.path_dataset = dataset_conf["path_dataset"]
        
        #read all file name from train/dev/test.txt files
        with open(self.filelist) as f:
            list_files = [line.rstrip() for line in f]
        self.game_files_name=list_files#[s + ".h5" for s in list_files]       
        
        if self.load_data_once4all:
            self.samples=np.zeros((len(self.game_files_name)*30,self.len_samples,8,8), dtype=int)
            self.outputs=np.zeros((len(self.game_files_name)*30,8*8), dtype=int)
            idx=0
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                h5f = h5py.File(self.path_dataset+gm_name,'r')
                game_log = np.array(h5f[gm_name.replace(".h5","")][:])
                h5f.close()
                last_board_state=copy.copy(game_log[0][-1])
                is_black_winner=isBlackWinner(game_log[1][-1],last_board_state)
                for sm_idx in range(30):
                    if is_black_winner:
                        end_move=2*sm_idx
                    else:
                        end_move=2*sm_idx+1
                        
                    if end_move+1 >= self.len_samples:
                        features=game_log[0][end_move-self.len_samples+1:
                                             end_move+1]
                    else:
                        features=[self.starting_board_stat]
                        #Padding starting board state before first index of sequence
                        for i in range(self.len_samples-end_move-2):
                            features.append(self.starting_board_stat)
                        #adding the inital of game as the end of sequence sample
                        for i in range(end_move+1):
                            features.append(game_log[0][i])

                    #if black is the current player the board should be multiplay by -1    
                    if is_black_winner:       
                        features=np.array([features],dtype=int)*-1
                    else:
                        features=np.array([features],dtype=int)    
                        
                    self.samples[idx]=features
                    self.outputs[idx]=np.array(game_log[1][end_move]).flatten()
                    idx+=1
        else:
        
            #create a list of samples as SampleManager objcets
            self.samples=np.empty(len(self.game_files_name)*30, dtype=object)
            idx=0
            for gm_idx,gm_name in tqdm(enumerate(self.game_files_name)):
                h5f = h5py.File(self.path_dataset+gm_name,'r')
                game_log = np.array(h5f[gm_name.replace(".h5","")][:])
                h5f.close()
                last_board_state=copy.copy(game_log[0][-1])
                is_black_winner=isBlackWinner(game_log[1][-1],last_board_state)
                for sm_idx in range(30):
                    if is_black_winner:
                        end_move=2*sm_idx
                    else:
                        end_move=2*sm_idx+1
                    self.samples[idx]=SampleManager(gm_name,
                                                    self.path_dataset,
                                                    end_move,
                                                    self.len_samples,
                                                    is_black_winner)
                    idx+=1
        
        #np.random.shuffle(self.samples)
        print(f"Number of samples: {len(self.samples)}")

        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        
        if self.load_data_once4all:
            features=self.samples[idx]
            y=self.outputs[idx]
        else:

            h5f = h5py.File(self.samples[idx].file_dir+self.samples[idx].game_name,'r')
            game_log = np.array(h5f[self.samples[idx].game_name.replace(".h5","")][:])
            h5f.close()

            if self.samples[idx].end_move+1 >= self.samples[idx].len_moves:
                features=game_log[0][self.samples[idx].end_move-self.samples[idx].len_moves+1:
                                     self.samples[idx].end_move+1]
            else:
                features=[self.starting_board_stat]
                #Padding starting board state before first index of sequence
                for i in range(self.samples[idx].len_moves-self.samples[idx].end_move-2):
                    features.append(self.starting_board_stat)
                #adding the inital of game as the end of sequence sample
                for i in range(self.samples[idx].end_move+1):
                    features.append(game_log[0][i])

            #if black is the current player the board should be multiplay by -1    
            if self.samples[idx].isBlackPlayer:       
                features=np.array([features],dtype=float)*-15
            else:
                features=np.array([features],dtype=float)

            #y is a move matrix
            y=np.array(game_log[1][self.samples[idx].end_move]).flatten()
            
        return features,y,self.len_samples
for optimizer in ["SGD", "Adam", "RMSprop", "Adagrad", "Adadelta"]:
    for learning_rate in [0.0001, 0.001, 0.01, 0.1, 1]:
        for batch_size in [100, 1000, 5000, 15000, 30000]:
            for epoch in [50, 100, 200, 500]:
                for hidden_dim in [128, 256]:
                    for activation_function in ["Linear", "ReLU", "Leaky ReLU", "Sigmoid", "Tanh"]:
                        conf={}
                        conf["path_save"]=f"saved_models/LSTM/{optimizer}/Learnings rate {learning_rate}/Batch size {batch_size}/Epoch {epoch}/{hidden_dim} {activation_function}"


                        if os.path.exists(conf["path_save"]):
                            continue

                        if torch.cuda.is_available():
                            #device = torch.device("cuda:0")
                            device = torch.device("cpu")
                        else:
                            device = torch.device("cpu")
                        print('Running on ' + str(device))

                        len_samples=5 # Could be modified?

                        dataset_conf={}  
                        # self.filelist : a list of all games for train/dev/test
                        dataset_conf["filelist"]="train.txt"
                        #len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
                        dataset_conf["len_samples"]=len_samples
                        dataset_conf["path_dataset"]="./dataset/"
                        dataset_conf['batch_size']=batch_size

                        print("Training Dataset... ")
                        ds_train = CustomDataset(dataset_conf)
                        trainSet = DataLoader(ds_train, batch_size=dataset_conf['batch_size']) # shuffle = True - Could be modified? For the best model

                        dataset_conf={}  
                        # self.filelist : a list of all games for train/dev/test
                        dataset_conf["filelist"]="dev.txt"
                        #len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
                        dataset_conf["len_samples"]=len_samples
                        dataset_conf["path_dataset"]="./dataset/"
                        dataset_conf['batch_size']=batch_size

                        print("Development Dataset... ")
                        ds_dev = CustomDataset(dataset_conf)
                        devSet = DataLoader(ds_dev, batch_size=dataset_conf['batch_size'])

                        conf["board_size"]=BOARD_SIZE
                        conf['epoch']=epoch
                        conf["earlyStopping"]=int(epoch/10)
                        conf["len_inpout_seq"]=len_samples
                        conf["LSTM_conf"]={}
                        conf["LSTM_conf"]["hidden_dim"]=hidden_dim
                        conf["activation_function"]=activation_function

                        model = LSTMs(conf).to(device)
                        if (optimizer == "Adam"):
                            opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
                        elif (optimizer == "SGD"):
                            opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
                        elif (optimizer == "Adagrad"):
                            opt = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
                        elif (optimizer == "Adadelta"):
                            opt = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
                        elif (optimizer == "RMSprop"):
                            opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

                        def count_parameters(model):
                            return sum(p.numel() for p in model.parameters() if p.requires_grad)

                        n = count_parameters(model)

                        print("Number of parameters: %s" % n)
                        os.makedirs(conf["path_save"], exist_ok=True)
                        f = open(f'{conf["path_save"]+" description"}.txt', 'a', encoding='utf-8')
                        f.write(f"Model: LSTM\nOptimizer: {optimizer}\nLearning rate: {learning_rate}\nHidden dimension layer: {hidden_dim}\nBatch size: {batch_size}\nEpoch: {epoch}\nEarlystopping: {int(epoch/10)}\nNumber of parameters: {n}\nThe best score on DEV : ")
                        f.close()

                        best_epoch=model.train_all(trainSet,
                                            devSet,
                                            conf['epoch'],
                                            device, opt)


    # model = torch.load(conf["path_save"] + '/model_2.pt')
    # model.eval()
    # train_clas_rep=model.evalulate(trainSet, device)
    # acc_train=train_clas_rep["weighted avg"]["recall"]
    # print(f"Accuracy Train: {round(100*acc_train,2)}%")
    # f = open('./saved_models/logs_save_models_new.txt', 'a', encoding='utf-8')
    # f.write(f"Accuracy Train: {round(100*acc_train,2)}%")
    # f.write("\n")
    # f.close()


