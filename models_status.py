import os

max_MLP = 0
max_LSTM = 0

for dropout in [0.1, 0.3, 0.5, 0.7]:
    for optimizer in ["SGD", "Adam", "RMSprop", "Adagrad", "Adadelta"]:
        for learning_rate in [0.0001, 0.001, 0.01, 0.1, 1]:
            for batch_size in [100, 1000, 5000, 15000, 30000]:
                for epoch in [50, 100, 200, 500]:
                    for hidden_dim in [128, 256]:
                        for activation_function in ["Linear", "ReLU", "Leaky ReLU", "Sigmoid", "Tanh"]:
                            conf={}
                            conf["path_save"]=f"saved_models/Dropout {dropout}/LSTM/{optimizer}/Learnings rate {learning_rate}/Batch size {batch_size}/Epoch {epoch}/{hidden_dim} {activation_function}"
                            if (os.path.exists(conf["path_save"])):
                                if (not os.path.exists(conf["path_save"]+" description.txt")):
                                    print("LOST ERROR")
                                    print(conf["path_save"])
                                if (len(os.listdir(conf["path_save"])) > 1):
                                    print(conf["path_save"])
                                if (len(os.listdir(conf["path_save"])) == 0):
                                    print("EMPTY ERROR")
                                    print(conf["path_save"])

                                f = open(f'{conf["path_save"]+" description"}.txt', encoding='utf-8')
                                for line in f.readlines():
                                    if ("DEV : " in line):
                                        if ("%" not in line):
                                            print("Uncompleted:")
                                            print(conf["path_save"])
                                        else: 
                                            number = float(line.split(": ")[1].split("%")[0])
                                            if (number > max_LSTM):
                                                max_LSTM = number
                                                path_max_LSTM = conf["path_save"]
                            if (os.path.exists(conf["path_save"]+" description.txt")):
                                if (not os.path.exists(conf["path_save"])):
                                    print("LOST ERROR")
                                    print(conf["path_save"])  
                                f.close()
                    for hidden_dim_1 in [128, 256]:
                        for hidden_dim_2 in [128, 256]:
                            for activation_function in ["Linear", "ReLU", "Leaky ReLU", "Sigmoid", "Tanh"]:
                                conf={}
                                conf["path_save"]=f"saved_models/Dropout {dropout}/MLP/{optimizer}/Learnings rate {learning_rate}/Batch size {batch_size}/Epoch {epoch}/{hidden_dim_1} {hidden_dim_2} {activation_function}"
                                if os.path.exists(conf["path_save"]):
                                    if (not os.path.exists(conf["path_save"]+" description.txt")):
                                        print("LOST ERROR")
                                        print(conf["path_save"])
                                    if (len(os.listdir(conf["path_save"])) > 1):
                                        print(conf["path_save"])
                                    if (len(os.listdir(conf["path_save"])) == 0):
                                        print("EMPTY ERROR")
                                        print(conf["path_save"])

                                    f = open(f'{conf["path_save"]+" description"}.txt', encoding='utf-8')
                                    for line in f.readlines():
                                        if ("DEV : " in line):
                                            if ("%" not in line):
                                                print("Uncompleted:")
                                                print(conf["path_save"])
                                            else: 
                                                number = float(line.split(": ")[1].split("%")[0])
                                                if (number > max_MLP):
                                                    max_MLP = number
                                                    path_max_MLP = conf["path_save"]
                                    f.close()
                                if (os.path.exists(conf["path_save"]+" description.txt")):
                                    if (not os.path.exists(conf["path_save"])):
                                        print("LOST ERROR")
                                        print(conf["path_save"])

print("\nThe best MLP model winrate: %.2f" % max_MLP + "%")
print(path_max_MLP)
print("The best LSTM model winrate: %.2f" % max_LSTM + "%")
print(path_max_LSTM)


