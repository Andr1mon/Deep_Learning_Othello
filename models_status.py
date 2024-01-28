import os
import matplotlib.pyplot as plt
import time

max_MLP_DEV = 0
max_LSTM_DEV = 0
total = 0
done = 0

dropout_list = [0.1, 0.3, 0.5, 0.7]
architecture_list = ["MLP", "LSTM"]
optimizer_list = ["SGD", "Adam", "RMSprop", "Adagrad", "Adadelta"]
learning_rate_list = [0.0001, 0.001, 0.01, 0.1, 1]
batch_size_list = [100, 1000, 5000, 15000, 30000]
epoch_list = [50, 100, 200, 300, 500]
hidden_dim_list = ["", " 128 ", " 96 ", " 192 ", " 256 "]
hidden_dim_list1 = ["256 ", "96 ", "128 ", "192 "]
activation_function_list = ["Leaky ReLU", "Linear", "ReLU",  "Sigmoid", "Tanh"]

for dropout in dropout_list:
    for architecture in architecture_list:
        for optimizer in optimizer_list:
            for learning_rate in learning_rate_list:
                for batch_size in batch_size_list:
                    for epoch in epoch_list:
                        for hidden_dim_1 in hidden_dim_list1:
                            for hidden_dim_2 in hidden_dim_list:
                                for hidden_dim_3 in hidden_dim_list:
                                    for hidden_dim_4 in hidden_dim_list:
                                        for hidden_dim_5 in hidden_dim_list:
                                            for activation_function1 in activation_function_list:
                                                for activation_function2 in activation_function_list:
                                                    for activation_function3 in activation_function_list:
                                                        for activation_function4 in activation_function_list:
                                                            for activation_function5 in activation_function_list:
                                                                conf={}
                                                                if (hidden_dim_2 == ""):
                                                                    hidden_dim_3 = ""
                                                                    hidden_dim_4 = ""
                                                                    hidden_dim_5 = ""
                                                                    activation_function2 = ""
                                                                    activation_function3 = ""
                                                                    activation_function4 = ""
                                                                    activation_function5 = ""
                                                                elif (hidden_dim_3 == ""):
                                                                    hidden_dim_4 = ""
                                                                    hidden_dim_5 = ""
                                                                    activation_function3 = ""
                                                                    activation_function4 = ""
                                                                    activation_function5 = ""
                                                                elif (hidden_dim_4 == ""):
                                                                    hidden_dim_5 = ""
                                                                    activation_function4 = ""
                                                                    activation_function5 = ""
                                                                elif (hidden_dim_5 == ""):
                                                                    activation_function5 = ""
                                                                conf["path_save"]=f"saved_models/Dropout {dropout}/{architecture}/{optimizer}/Learnings rate {learning_rate}/Batch size {batch_size}/Epoch {epoch}/{hidden_dim_1}{activation_function1}{hidden_dim_2}{activation_function2}{hidden_dim_3}{activation_function3}{hidden_dim_4}{activation_function4}{hidden_dim_5}{activation_function5}"
                                                                                                
                                                                
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
                                                                                if (architecture == "LSTM" and number > max_LSTM_DEV):
                                                                                    max_LSTM_DEV = number
                                                                                    path_max_LSTM_DEV = conf["path_save"]
                                                                                elif (architecture == "MLP" and number > max_MLP_DEV):
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
                                                                                    plt.plot(epochs, train_accuracy, 'b-', epochs, dev_accuracy, 'r-')
                                                                                    plt.xlabel('Epochs')
                                                                                    plt.ylabel('Accuracy (%)')
                                                                                    plt.legend(('Train', 'Validation'), shadow=True)
                                                                                    plt.title('Training and Validation Accuracy')
                                                                                    plt.savefig(f"{conf['path_save']} curve.png")
                                                                                    plt.clf()
                                                                                    #plt.show()
                                                                    f.close()
                                                                if (os.path.exists(conf["path_save"]+" description.txt")):
                                                                    if (not os.path.exists(conf["path_save"])):
                                                                        print("LOST ERROR")
                                                                        print(conf["path_save"])  

print("****************************************************************************")
print("The best MLP accuracy on DEV: %.2f" % max_MLP_DEV + "%")
print(path_max_MLP_DEV)
print("The best LSTM accuracy on DEV: %.2f" % max_LSTM_DEV + "%")
print(path_max_LSTM_DEV)
print("****************************************************************************")
print(f"Total: {total} Done: {done}")


