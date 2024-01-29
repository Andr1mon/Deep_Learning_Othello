import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import copy
import time
import matplotlib.pyplot as plt
from datetime import datetime
import h5py
import torch
import torch.nn as nn

from utile import get_legal_moves,is_legal_move,has_tile_to_flip,initialze_board

BOARD_SIZE=8


def input_seq_generator(board_stats_seq,length_seq):
    
    board_stat_init=initialze_board()

    if len(board_stats_seq) >= length_seq:
        input_seq=board_stats_seq[-length_seq:]
    else:
        input_seq=[board_stat_init]
        #Padding starting board state before first index of sequence
        for i in range(length_seq-len(board_stats_seq)-1):
            input_seq.append(board_stat_init)
        #adding the inital of game as the end of sequence sample
        for i in range(len(board_stats_seq)):
            input_seq.append(board_stats_seq[i])
            
    return input_seq

def find_best_move(move1_prob,legal_moves):
    """
    Finds the best move based on the provided move probabilities and legal moves.

    Parameters:
    - move1_prob (numpy.ndarray): 2D array representing the probabilities of moves.
    - legal_moves (list): List of legal moves.

    Returns:
    - tuple: The best move coordinates (row, column).
    """

    # Initialize the best move with the first legal move
    best_move=legal_moves[0]
    
    # Initialize the maximum score with the probability of the first legal move
    max_score=move1_prob[legal_moves[0][0],legal_moves[0][1]]
    
    # Iterate through all legal moves to find the one with the maximum probability
    for i in range(len(legal_moves)):
        # Update the best move if the current move has a higher probability
        if move1_prob[legal_moves[i][0],legal_moves[i][1]]>max_score:
            max_score=move1_prob[legal_moves[i][0],legal_moves[i][1]]
            best_move=legal_moves[i]
    return best_move

def apply_flip(best_move,board_stat,NgBlackPsWhith):
    """
    Apply tile flipping on the Othello board based on the best move.

    Parameters:
    - best_move (tuple): Coordinates (row, column) of the best move.
    - board_stat (numpy.ndarray): 2D array representing the current state of the Othello board.
    - NgBlackPsWhith (int): Indicator for the current player (Black: -1, White: 1).

    Returns:
    - numpy.ndarray: Updated Othello board after applying tile flipping.
    """
    
    MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1),
             (0, -1),           (0, +1),
             (+1, -1), (+1, 0), (+1, +1)]

    for direction in MOVE_DIRS:
        if has_tile_to_flip(best_move, direction,board_stat,NgBlackPsWhith):
            i = 1
            while True:
                row = best_move[0] + direction[0] * i
                col = best_move[1] + direction[1] * i
                if board_stat[row][col] == board_stat[best_move[0], best_move[1]]:
                    break
                else:
                    board_stat[row][col] = board_stat[best_move[0], best_move[1]]
                    i += 1
                    
    return board_stat

def start_game(conf):
    for g in [0,1]:
        # Two rounds of game would be played
        # First player1 starts game, and then Player2 starts the other game
        if g:
            conf['player1'], conf['player2'] = conf['player2'], conf['player1']
        
        #print(conf['player1'])
        #print("VERSUS")
        #print(conf['player2'])
        data = np.zeros((2,60,8,8))
        k = 0
        board_stat=initialze_board()
        moves_log=""
        board_stats_seq=[]
        pass2player=False

        while not np.all(board_stat) and not pass2player:
            NgBlackPsWhith=-1
            board_stats_seq.append(copy.copy(board_stat))
            model = torch.load(conf['player1'],map_location=torch.device('cpu'))
            model.eval()

            input_seq_boards=input_seq_generator(board_stats_seq,model.len_inpout_seq)
            #if black is the current player the board should be multiplay by -1
            model_input=np.array([input_seq_boards])*-1
            move1_prob = model(torch.tensor(model_input).float().to(device))
            move1_prob=move1_prob.cpu().detach().numpy().reshape(8,8)

            legal_moves=get_legal_moves(board_stat,NgBlackPsWhith)

            if len(legal_moves)>0:
                
                best_move=find_best_move(move1_prob,legal_moves)
                #print(f"Black: {best_move} < from possible move {legal_moves}")
                data[0][k]=board_stat
                data[1][k][best_move[0]][best_move[1]] = 1
                k+=1
                board_stat[best_move[0],best_move[1]]=NgBlackPsWhith
                moves_log+=str(best_move[0]+1)+str(best_move[1]+1)
                
                board_stat=apply_flip(best_move,board_stat,NgBlackPsWhith)
                
                
            else:
                #print("Black pass")
                if moves_log[-2:]=="__":
                    pass2player=True
                moves_log+="__"


            NgBlackPsWhith=+1
            board_stats_seq.append(copy.copy(board_stat))
            model = torch.load(conf['player2'])
            model = torch.load(conf['player2'],map_location=torch.device('cpu'))
            model.eval()

            input_seq_boards=input_seq_generator(board_stats_seq,model.len_inpout_seq)
            #if black is the current player the board should be multiplay by -1
            model_input=np.array([input_seq_boards])
            move1_prob = model(torch.tensor(model_input).float().to(device))
            move1_prob=move1_prob.cpu().detach().numpy().reshape(8,8)

            legal_moves=get_legal_moves(board_stat,NgBlackPsWhith)


            if len(legal_moves)>0:
                
                best_move = find_best_move(move1_prob,legal_moves)
                #print(f"White: {best_move} < from possible move {legal_moves}")
                data[0][k]=board_stat
                data[1][k][best_move[0]][best_move[1]] = 1
                k+=1
                board_stat[best_move[0],best_move[1]]=NgBlackPsWhith
                moves_log+=str(best_move[0]+1)+str(best_move[1]+1)
                board_stat=apply_flip(best_move,board_stat,NgBlackPsWhith)
            else:
                #print("White pass")
                if moves_log[-2:]=="__":
                    pass2player=True
                moves_log+="__"
        board_stats_seq.append(copy.copy(board_stat))
        
        #print("Moves log:",moves_log)
        
        
        if np.sum(board_stat)<0:
            #print(f"Black {conf['player1']} is winner (with {-1*int(np.sum(board_stat))} points)")
            if not g:
                conf['wins'] += 1
        elif np.sum(board_stat)>0:
            #print(f"White {conf['player2']} is winner (with {int(np.sum(board_stat))} points)")
            if g:
                conf['wins'] += 1
        #else:
            #print(f"Draw")

        #save game log in gif file
        """
        fig,ax = plt.subplots()
        ims = []
        for i in range(len(board_stats_seq)):
            im = plt.imshow(board_stats_seq[i]*-1,
                            extent=[0.5,8.5,0.5,8.5],
                            cmap='binary',
                            interpolation='nearest')
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)   
        ani.save(f"games/game_{g}.gif", writer='imagemagick', fps=0.8)
        """
        with h5py.File(f"AI_dataset/{str(datetime.now()).replace('.','-').replace(':','-')}.h5", 'w') as h5f:
            h5f.create_dataset('dataset', data=data)
            """
            for i in range (60):
                np.savetxt(f"AI_dataset_status/{str(datetime.now()).replace('.','-').replace(':','-')}.txt", data[0][i])
                np.savetxt(f"AI_dataset_moves/{str(datetime.now()).replace('.','-').replace(':','-')}.txt", data[1][i])
            """
    conf['player1'], conf['player2'] = conf['player2'], conf['player1']
    conf['games'] += 2

        
    

if torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

conf={}
for dropout1 in os.listdir("saved_models"):
    for architecture1 in os.listdir(f'saved_models/{dropout1}'):
        for optimizer1 in os.listdir(f'saved_models/{dropout1}/{architecture1}'):
            for learning_rate1 in os.listdir(f'saved_models/{dropout1}/{architecture1}/{optimizer1}'):
                for batch_size1 in os.listdir(f'saved_models/{dropout1}/{architecture1}/{optimizer1}/{learning_rate1}'):
                    for epoch1 in os.listdir(f'saved_models/{dropout1}/{architecture1}/{optimizer1}/{learning_rate1}/{batch_size1}'):
                        for layers1 in os.listdir(f'saved_models/{dropout1}/{architecture1}/{optimizer1}/{learning_rate1}/{batch_size1}/{epoch1}'):
                            conf["path_save"]=f"saved_models/{dropout1}/{architecture1}/{optimizer1}/{learning_rate1}/{batch_size1}/{epoch1}/{layers1}"
                            if ("description" in conf["path_save"] or "logs" in conf["path_save"] or "curve" in conf["path_save"]):
                                continue
                            conf['games']=0
                            conf['wins']=0
                            conf['player1']=conf["path_save"]+'//'+os.listdir(conf["path_save"])[0]
                            print(conf['player1'])
                            for dropout2 in os.listdir("saved_models"):
                                for architecture2 in os.listdir(f'saved_models/{dropout2}'):
                                    for optimizer2 in os.listdir(f'saved_models/{dropout2}/{architecture2}'):
                                        for learning_rate2 in os.listdir(f'saved_models/{dropout2}/{architecture2}/{optimizer2}'):
                                            for batch_size2 in os.listdir(f'saved_models/{dropout2}/{architecture2}/{optimizer2}/{learning_rate2}'):
                                                for epoch2 in os.listdir(f'saved_models/{dropout2}/{architecture2}/{optimizer2}/{learning_rate2}/{batch_size2}'):
                                                    for layers2 in os.listdir(f'saved_models/{dropout2}/{architecture2}/{optimizer2}/{learning_rate2}/{batch_size2}/{epoch2}'):
                                                        conf["path_save"]=f"saved_models/{dropout2}/{architecture2}/{optimizer2}/{learning_rate2}/{batch_size2}/{epoch2}/{layers2}"
                                                        if ("description" in conf["path_save"] or "logs" in conf["path_save"] or "curve" in conf["path_save"]):
                                                            continue
                                                        conf["path_save"]=f"saved_models/{dropout2}/{architecture2}/{optimizer2}/{learning_rate2}/{batch_size2}/{epoch2}/{layers2}"
                                                        conf['player2']=conf["path_save"]+'//'+os.listdir(conf["path_save"])[0]
                                                        print(conf['games'], conf['wins'])
                            print(f"Games: {conf['games']}\nWins: {conf['wins']}\nWinrate: {100*conf['wins']/conf['games']}%")
                            """
                            f = open(f'{conf["path_save"]} description.txt', 'a', encoding='utf-8')
                            f.write(f"\nWinrate: {100*conf['wins']/conf['games']}%")
                            f.close()
                            """


