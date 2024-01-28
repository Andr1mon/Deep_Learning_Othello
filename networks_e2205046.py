import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import h5py
import json
import copy
import time
from datetime import datetime


def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets) # Could be improved?


class MLP(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP, self).__init__()  
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.hidden_dim_1=conf["MLP_conf"]["hidden_dim_1"]
        self.hidden_dim_2=conf["MLP_conf"]["hidden_dim_2"]
        self.hidden_dim_3=conf["MLP_conf"]["hidden_dim_3"]
        self.hidden_dim_4=conf["MLP_conf"]["hidden_dim_4"]
        self.hidden_dim_5=conf["MLP_conf"]["hidden_dim_5"]
        self.activation_function1=conf["activation_function1"]
        self.activation_function2=conf["activation_function2"]
        self.activation_function3=conf["activation_function3"]
        self.activation_function4=conf["activation_function4"]
        self.activation_function5=conf["activation_function5"]



        # Define the layers of the MLP
        if (self.activation_function1 == "ReLU"):
            self.act_function1 = nn.ReLU()
        elif (self.activation_function1 == "Sigmoid"):
            self.act_function1 = nn.Sigmoid()
        elif (self.activation_function1 == "Tanh"):
            self.act_function1 = nn.Tanh()
        elif (self.activation_function1 == "Leaky ReLU"):
            self.act_function1 = nn.LeakyReLU()
        if (self.activation_function2 == "ReLU"):
            self.act_function2 = nn.ReLU()
        elif (self.activation_function2 == "Sigmoid"):
            self.act_function2 = nn.Sigmoid()
        elif (self.activation_function2 == "Tanh"):
            self.act_function2 = nn.Tanh()
        elif (self.activation_function2 == "Leaky ReLU"):
            self.act_function2 = nn.LeakyReLU()
        if (self.activation_function3 == "ReLU"):
            self.act_function3 = nn.ReLU()
        elif (self.activation_function3 == "Sigmoid"):
            self.act_function3 = nn.Sigmoid()
        elif (self.activation_function3 == "Tanh"):
            self.act_function3 = nn.Tanh()
        elif (self.activation_function3 == "Leaky ReLU"):
            self.act_function3 = nn.LeakyReLU()
        if (self.activation_function4 == "ReLU"):
            self.act_function4 = nn.ReLU()
        elif (self.activation_function4 == "Sigmoid"):
            self.act_function4 = nn.Sigmoid()
        elif (self.activation_function4 == "Tanh"):
            self.act_function4 = nn.Tanh()
        elif (self.activation_function4 == "Leaky ReLU"):
            self.act_function4 = nn.LeakyReLU()
        if (self.activation_function5 == "ReLU"):
            self.act_function5 = nn.ReLU()
        elif (self.activation_function5 == "Sigmoid"):
            self.act_function5 = nn.Sigmoid()
        elif (self.activation_function5 == "Tanh"):
            self.act_function5 = nn.Tanh()
        elif (self.activation_function5 == "Leaky ReLU"):
            self.act_function5 = nn.LeakyReLU()

        self.hidden_dim_1 = int(self.hidden_dim_1)
        self.lin1 = nn.Linear(self.board_size*self.board_size, self.hidden_dim_1)
        if (self.hidden_dim_2 == ""):
            self.lin2 = nn.Linear(self.hidden_dim_1, self.board_size*self.board_size)
        elif (self.hidden_dim_3 == ""):
            self.hidden_dim_2 = int(self.hidden_dim_2)
            self.lin2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
            self.lin3 = nn.Linear(self.hidden_dim_2, self.board_size*self.board_size)
        elif (self.hidden_dim_4 == ""):
            self.hidden_dim_2 = int(self.hidden_dim_2)
            self.hidden_dim_3 = int(self.hidden_dim_3)
            self.lin2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
            self.lin3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_3)
            self.lin4 = nn.Linear(self.hidden_dim_3, self.board_size*self.board_size)
        elif (self.hidden_dim_5 == ""):
            self.hidden_dim_2 = int(self.hidden_dim_2)
            self.hidden_dim_3 = int(self.hidden_dim_3)
            self.hidden_dim_4 = int(self.hidden_dim_4)
            self.lin2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
            self.lin3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_3)
            self.lin4 = nn.Linear(self.hidden_dim_3, self.hidden_dim_4)
            self.lin5 = nn.Linear(self.hidden_dim_4, self.board_size*self.board_size)
        else:
            self.hidden_dim_2 = int(self.hidden_dim_2)
            self.hidden_dim_3 = int(self.hidden_dim_3)
            self.hidden_dim_4 = int(self.hidden_dim_4)
            self.hidden_dim_5 = int(self.hidden_dim_5)
            self.lin2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
            self.lin3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_3)
            self.lin4 = nn.Linear(self.hidden_dim_3, self.hidden_dim_4)
            self.lin5 = nn.Linear(self.hidden_dim_4, self.hidden_dim_5)
            self.lin6 = nn.Linear(self.hidden_dim_5, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=conf["dropout"])
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        if (self.activation_function1 != "Linear"):
            x = self.act_function1(x)
        x = self.lin2(x)
    
        if (self.hidden_dim_2 == ""):
            outp = x
        elif (self.hidden_dim_3 == ""):
            if (self.activation_function2 != "Linear"):
                x = self.act_function2(x)
            outp = self.lin3(x)
        elif (self.hidden_dim_4 == ""):
            if (self.activation_function2 != "Linear"):
                x = self.act_function2(x)
            x = self.lin3(x)
            if (self.activation_function3 != "Linear"):
                x = self.act_function3(x)
            outp = self.lin4(x)
        elif (self.hidden_dim_5 == ""):
            if (self.activation_function2 != "Linear"):
                x = self.act_function2(x)
            x = self.lin3(x)
            if (self.activation_function3 != "Linear"):
                x = self.act_function3(x)
            x = self.lin4(x)
            if (self.activation_function4 != "Linear"):
                x = self.act_function4(x)
            outp = self.lin5(x)
        else:
            if (self.activation_function2 != "Linear"):
                x = self.act_function2(x)
            x = self.lin3(x)
            if (self.activation_function3 != "Linear"):
                x = self.act_function3(x)
            x = self.lin4(x)
            if (self.activation_function4 != "Linear"):
                x = self.act_function4(x)
            x = self.lin5(x)
            if (self.activation_function5 != "Linear"):
                x = self.act_function5(x)
            outp = self.lin6(x)
        
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0 # to manage earlystopping
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch: " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            f = open(f'{self.path_save} logs.txt', 'a', encoding='utf-8')
            f.write("epoch: " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            f.write("\n")
            f.close()
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train: {round(100*acc_train,2)}%, Dev: {round(100*acc_dev,2)}% ;",
                  f"Time: {round(time.time()-init_time)}",
                  f"(last_train: {round(last_training)}sec, last_pred: {round(last_prediction)}sec)")
            f = open(f'{self.path_save} logs.txt', 'a', encoding='utf-8')
            f.write(f"Accuracy Train: {round(100*acc_train,2)}%, Dev: {round(100*acc_dev,2)}% ;" +
                  f" Time: {round(time.time()-init_time)}" +
                  f" (last_train: {round(last_training)}sec, last_pred: {round(last_prediction)}sec)")
            f.write("\n")
            f.close()

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                for filename in os.listdir(self.path_save):
                    os.remove(self.path_save+'/'+filename)
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch}: {round(100*best_dev,3)}%")
            f = open(f'{self.path_save} logs.txt', 'a', encoding='utf-8')
            f.write("*"*15 + f" The best score on DEV {best_epoch}: {round(100*best_dev,3)}%")
            f.write("\n")
            f.close()

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc: {100*_clas_rep['weighted avg']['recall']}%")
        f = open(f'{self.path_save} logs.txt', 'a', encoding='utf-8')
        f.write(f"Recalculing the best DEV: WAcc: {100*_clas_rep['weighted avg']['recall']}%")
        f.write("\n")
        f.close()
        f = open(f'{self.path_save} description.txt', 'a', encoding='utf-8')
        f.write(f"{100*_clas_rep['weighted avg']['recall']}%")
        f.close()

        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target,_ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().detach().numpy()
            target=target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep
    
    

class LSTMs(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMs, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.hidden_dim_1=conf["LSTM_conf"]["hidden_dim_1"]
        self.hidden_dim_2=conf["LSTM_conf"]["hidden_dim_2"]
        self.hidden_dim_3=conf["LSTM_conf"]["hidden_dim_3"]
        self.hidden_dim_4=conf["LSTM_conf"]["hidden_dim_4"]
        self.hidden_dim_5=conf["LSTM_conf"]["hidden_dim_5"]
        self.activation_function1=conf["activation_function1"]
        self.activation_function2=conf["activation_function2"]
        self.activation_function3=conf["activation_function3"]
        self.activation_function4=conf["activation_function4"]
        self.activation_function5=conf["activation_function5"]

        # Define the layers of the LSTM model
        if (self.activation_function1 == "ReLU"):
            self.act_function1 = nn.ReLU()
        elif (self.activation_function1 == "Sigmoid"):
            self.act_function1 = nn.Sigmoid()
        elif (self.activation_function1 == "Tanh"):
            self.act_function1 = nn.Tanh()
        elif (self.activation_function1 == "Leaky ReLU"):
            self.act_function1 = nn.LeakyReLU()
        if (self.activation_function2 == "ReLU"):
            self.act_function2 = nn.ReLU()
        elif (self.activation_function2 == "Sigmoid"):
            self.act_function2 = nn.Sigmoid()
        elif (self.activation_function2 == "Tanh"):
            self.act_function2 = nn.Tanh()
        elif (self.activation_function2 == "Leaky ReLU"):
            self.act_function2 = nn.LeakyReLU()
        if (self.activation_function3 == "ReLU"):
            self.act_function3 = nn.ReLU()
        elif (self.activation_function3 == "Sigmoid"):
            self.act_function3 = nn.Sigmoid()
        elif (self.activation_function3 == "Tanh"):
            self.act_function3 = nn.Tanh()
        elif (self.activation_function3 == "Leaky ReLU"):
            self.act_function3 = nn.LeakyReLU()
        if (self.activation_function4 == "ReLU"):
            self.act_function4 = nn.ReLU()
        elif (self.activation_function4 == "Sigmoid"):
            self.act_function4 = nn.Sigmoid()
        elif (self.activation_function4 == "Tanh"):
            self.act_function4 = nn.Tanh()
        elif (self.activation_function4 == "Leaky ReLU"):
            self.act_function4 = nn.LeakyReLU()
        if (self.activation_function5 == "ReLU"):
            self.act_function5 = nn.ReLU()
        elif (self.activation_function5 == "Sigmoid"):
            self.act_function5 = nn.Sigmoid()
        elif (self.activation_function5 == "Tanh"):
            self.act_function5 = nn.Tanh()
        elif (self.activation_function5 == "Leaky ReLU"):
            self.act_function5 = nn.LeakyReLU()

        self.hidden_dim_1 = int(self.hidden_dim_1)
        self.lstm = nn.LSTM(self.board_size*self.board_size, self.hidden_dim_1, batch_first=True)
        
        if (self.hidden_dim_2 == ""):
            self.lin2 = nn.Linear(self.hidden_dim_1, self.board_size*self.board_size)
        elif (self.hidden_dim_3 == ""):
            self.hidden_dim_2 = int(self.hidden_dim_2)
            self.lin2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
            self.lin3 = nn.Linear(self.hidden_dim_2, self.board_size*self.board_size)
        elif (self.hidden_dim_4 == ""):
            self.hidden_dim_2 = int(self.hidden_dim_2)
            self.hidden_dim_3 = int(self.hidden_dim_3)
            self.lin2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
            self.lin3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_3)
            self.lin4 = nn.Linear(self.hidden_dim_3, self.board_size*self.board_size)
        elif (self.hidden_dim_5 == ""):
            self.hidden_dim_2 = int(self.hidden_dim_2)
            self.hidden_dim_3 = int(self.hidden_dim_3)
            self.hidden_dim_4 = int(self.hidden_dim_4)
            self.lin2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
            self.lin3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_3)
            self.lin4 = nn.Linear(self.hidden_dim_3, self.hidden_dim_4)
            self.lin5 = nn.Linear(self.hidden_dim_4, self.board_size*self.board_size)
        else:
            self.hidden_dim_2 = int(self.hidden_dim_2)
            self.hidden_dim_3 = int(self.hidden_dim_3)
            self.hidden_dim_4 = int(self.hidden_dim_4)
            self.hidden_dim_5 = int(self.hidden_dim_5)
            self.lin2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
            self.lin3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_3)
            self.lin4 = nn.Linear(self.hidden_dim_3, self.hidden_dim_4)
            self.lin5 = nn.Linear(self.hidden_dim_4, self.hidden_dim_5)
            self.lin6 = nn.Linear(self.hidden_dim_5, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=conf["dropout"])

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        x, (hn, cn) = self.lstm(seq)
        
        #using output sequence as below 
        #(lstm_out[:,-1,:] pass only last vector of output sequence)
        if len(seq.shape)>2: # to manage the batch of sample
            # Training phase where input is batch of seq
            if (self.activation_function1 != "Linear"):
                x = self.act_function1(x[:,-1,:])
                x = self.lin2(x)
            else:
                x = self.lin2(x[:,-1,:])
        else:
            # Prediction phase where input is a single seq
            if (self.activation_function1 != "Linear"):
                x = self.act_function1(x[-1,:])
                x = self.lin2(x)
            else:
                x = self.lin2(x[-1,:])
        
        if (self.hidden_dim_2 == ""):
            outp = x
        elif (self.hidden_dim_3 == ""):
            if (self.activation_function2 != "Linear"):
                x = self.act_function2(x)
            outp = self.lin3(x)
        elif (self.hidden_dim_4 == ""):
            if (self.activation_function2 != "Linear"):
                x = self.act_function2(x)
            x = self.lin3(x)
            if (self.activation_function3 != "Linear"):
                x = self.act_function3(x)
            outp = self.lin4(x)
        elif (self.hidden_dim_5 == ""):
            if (self.activation_function2 != "Linear"):
                x = self.act_function2(x)
            x = self.lin3(x)
            if (self.activation_function3 != "Linear"):
                x = self.act_function3(x)
            x = self.lin4(x)
            if (self.activation_function4 != "Linear"):
                x = self.act_function4(x)
            outp = self.lin5(x)
        else:
            if (self.activation_function2 != "Linear"):
                x = self.act_function2(x)
            x = self.lin3(x)
            if (self.activation_function3 != "Linear"):
                x = self.act_function3(x)
            x = self.lin4(x)
            if (self.activation_function4 != "Linear"):
                x = self.act_function4(x)
            x = self.lin5(x)
            if (self.activation_function5 != "Linear"):
                x = self.act_function5(x)
            outp = self.lin6(x)

        if len(seq.shape)>2:
            outp = F.softmax(outp, dim=1).squeeze()
        else:  
            outp = F.softmax(outp, dim=-1).squeeze()
            
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch: " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            f = open(f'{self.path_save} logs.txt', 'a', encoding='utf-8')
            f.write("epoch: " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            f.write("\n")
            f.close()
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train: {round(100*acc_train,2)}%, Dev: {round(100*acc_dev,2)}% ;",
                  f"Time: {round(time.time()-init_time)}",
                  f"(last_train: {round(last_training)}, last_pred: {round(last_prediction)})")
            f = open(f'{self.path_save} logs.txt', 'a', encoding='utf-8')
            f.write(f"Accuracy Train: {round(100*acc_train,2)}%, Dev: {round(100*acc_dev,2)}% ; Time: {round(time.time()-init_time)} (last_train: {round(last_training)}, last_pred: {round(last_prediction)})")
            f.write("\n")
            f.close()

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                for filename in os.listdir(self.path_save):
                    os.remove(self.path_save+'/'+filename)
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch}: {round(100*best_dev,3)}%")
            f = open(f'{self.path_save} logs.txt', 'a', encoding='utf-8')
            f.write("*"*15 + f" The best score on DEV {best_epoch}: {round(100*best_dev,3)}%")
            f.write("\n")
            f.close()

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc: {100*_clas_rep['weighted avg']['recall']}%")
        f = open(f'{self.path_save} logs.txt', 'a', encoding='utf-8')
        f.write(f"Recalculing the best DEV: WAcc: {100*_clas_rep['weighted avg']['recall']}%")
        f.write("\n")
        f.close()
        f = open(f'{self.path_save} description.txt', 'a', encoding='utf-8')
        f.write(f"{100*_clas_rep['weighted avg']['recall']}%")
        f.close()

        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target_array,lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().clone().detach().numpy()
            target=target_array.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep
            

