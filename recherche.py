dropout_list = [0.1]
optimizer_list = ["Adam"]
learning_rate_list = [0.0001, 0.001, 0.005, 0.01, 0.1, 1]
batch_size_list = [1000]
epoch_list = [200]
hidden_dim_list = [" 128 "]
hidden_dim_list1 = ["128 "]
activation_function_list = ["Linear"]


for dropout in dropout_list:
    for optimizer in optimizer_list:
        for learning_rate in learning_rate_list:
            for batch_size in batch_size_list:
                for epoch in epoch_list:
                    for hidden_dim_1 in hidden_dim_list1:
                        for hidden_dim_2 in hidden_dim_list:
                            for hidden_dim_3 in [""]:
                                for hidden_dim_4 in [""]:
                                    for hidden_dim_5 in [""]:
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
                                                            conf["path_save"]=f"saved_models/Dropout {dropout}/MLP/{optimizer}/Learnings rate {learning_rate}/Batch size {batch_size}/Epoch {epoch}/{hidden_dim_1}{activation_function1}{hidden_dim_2}{activation_function2}{hidden_dim_3}{activation_function3}{hidden_dim_4}{activation_function4}{hidden_dim_5}{activation_function5}"
                                                            print(conf['path_save'], 'description.txt')

dropout_list = [0.1]
optimizer_list = ["Adam"]
learning_rate_list = [0.0001, 0.001, 0.005, 0.01, 0.1, 1]
batch_size_list = [1000]
epoch_list = [200]
hidden_dim_list = [" 128 "]
hidden_dim_list1 = ["128 "]
activation_function_list = ["Linear"]


for dropout in dropout_list:
    for optimizer in optimizer_list:
        for learning_rate in learning_rate_list:
            for batch_size in batch_size_list:
                for epoch in epoch_list:
                    for hidden_dim_1 in hidden_dim_list1:
                        for hidden_dim_2 in [""]:
                            for hidden_dim_3 in [""]:
                                for hidden_dim_4 in [""]:
                                    for hidden_dim_5 in [""]:
                                        for activation_function1 in activation_function_list:
                                            for activation_function2 in activation_function_list:
                                                for activation_function3 in activation_function_list:
                                                    for activation_function4 in activation_function_list:
                                                        for activation_function5 in activation_function_list:
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
                                                            conf={}
                                                            conf["path_save"]=f"saved_models/Dropout {dropout}/LSTM/{optimizer}/Learnings rate {learning_rate}/Batch size {batch_size}/Epoch {epoch}/{hidden_dim_1}{activation_function1}{hidden_dim_2}{activation_function2}{hidden_dim_3}{activation_function3}{hidden_dim_4}{activation_function4}{hidden_dim_5}{activation_function5}"
                                                            print(conf['path_save'], 'description.txt')
