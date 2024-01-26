import os
import matplotlib.pyplot as plt
import time

max_MLP_DEV = 0
max_LTSM_DEV = 0
total = 0
done = 0


for dropout in [0.1, 0.3, 0.5, 0.7]:
    for optimizer in ["SGD", "Adam", "RMSprop", "Adagrad", "Adadelta"]:
        for learning_rate in [0.0001, 0.001, 0.01, 0.1, 1]:
            for batch_size in [100, 1000, 5000, 15000, 30000]:
                for epoch in [50, 100, 200, 300, 500]:
                    for hidden_dim in [96, 128, 192, 256]:
                        for activation_function in ["Linear", "ReLU", "Leaky ReLU", "Sigmoid", "Tanh"]:
                            conf={}
                            conf["path_save"]=f"saved_models/Dropout {dropout}/LSTM/{optimizer}/Learnings rate {learning_rate}/Batch size {batch_size}/Epoch {epoch}/{hidden_dim} {activation_function}"
                            epochs = []
                            train_accuracy = []
                            dev_accuracy = []
                            total += 1
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
                                            done += 1
                                            if (number > max_LTSM_DEV):
                                                max_LTSM_DEV = number
                                                path_max_LTSM_DEV = conf["path_save"]
                            if (os.path.exists(conf["path_save"]+" description.txt")):
                                if (not os.path.exists(conf["path_save"])):
                                    print("LOST ERROR")
                                    print(conf["path_save"])  
                                f.close()
                    for hidden_dim_1 in [96, 128, 192, 256]:
                        for hidden_dim_2 in [96, 128, 192, 256]:
                            for activation_function in ["Linear", "ReLU", "Leaky ReLU", "Sigmoid", "Tanh"]:
                                conf={}
                                conf["path_save"]=f"saved_models/Dropout {dropout}/MLP/{optimizer}/Learnings rate {learning_rate}/Batch size {batch_size}/Epoch {epoch}/{hidden_dim_1} {hidden_dim_2} {activation_function}"
                                epochs = []
                                train_accuracy = []
                                dev_accuracy = []
                                total += 1
                                if os.path.exists(conf["path_save"]):
                                    if (not os.path.exists(conf["path_save"]+" description.txt")):
                                        print("LOST ERROR")
                                        print(conf["path_save"])
                                    if (len(os.listdir(conf["path_save"])) > 1):
                                        print("MODELS WARNING")
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
                                                done += 1
                                                if (number > max_MLP_DEV):
                                                    max_MLP_DEV = number
                                                    path_max_MLP_DEV = conf["path_save"]
                                                if (not os.path.exists(conf["path_save"]+" curve.png")):
                                                    g = open(f'{conf["path_save"]} logs.txt', encoding='utf-8')
                                                    for line in g.readlines():
                                                        if ('epoch: ' in line):
                                                            epochs.append(int(line.split('epoch: ')[1].split('/')[0]))
                                                        if ('Accuracy Train: ' in line):
                                                            train_accuracy.append(float(line.split('Accuracy Train: ')[1].split('%')[0]))
                                                            dev_accuracy.append(float(line.split('Dev: ')[1].split('%')[0]))
                                                    g.close()
                                                    """
                                                    if (len(epochs) >= 300):
                                                        plt.plot(epochs, train_accuracy, 'b-', epochs, dev_accuracy, 'r-')
                                                        plt.xlabel('Epochs')
                                                        plt.ylabel('Accuracy')
                                                        plt.show()
                                                        time.sleep(1000000)
                                                    """
                                                

                                    f.close()
                                if (os.path.exists(conf["path_save"]+" description.txt")):
                                    if (not os.path.exists(conf["path_save"])):
                                        print("LOST ERROR")
                                        print(conf["path_save"])


print("****************************************************************************")
print("The best MLP accuracy on DEV: %.2f" % max_MLP_DEV + "%")
print(path_max_MLP_DEV)
print("The best LSTM accuracy on DEV: %.2f" % max_LTSM_DEV + "%")
print(path_max_LTSM_DEV)
print("****************************************************************************")
print(f"Total: {total} Done: {done}")


