import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys

def example_func():
    #backprop_example1.txt
    theta_weights = [[[0.40000,  0.10000 ], [0.30000,  0.20000]], [[0.70000,  0.50000,  0.60000]]]
    train_list = [[0.13000], [0.42000]]
    y_train = [[0.90000], [0.23000]]
    alpha = 0.1
    #backprop_example2.txt
    # theta_weights = [[[0.42000,  0.15000,  0.40000], [0.72000,  0.10000,  0.54000], [0.01000,  0.19000,  0.42000 ], [0.30000,  0.35000,  0.68000 ]], [[0.21000,  0.67000,  0.14000,  0.96000,  0.87000], [0.87000,  0.42000,  0.20000,  0.32000, 0.89000], [0.03000,  0.56000,  0.80000, 0.69000,  0.09000 ]], [[0.04000,  0.87000,  0.42000,  0.53000], [0.17000,  0.10000,  0.95000,  0.69000 ]]]
    # train_list = [[0.32000,  0.68000], [0.83000,   0.02000]]
    # y_train = [[0.75000,   0.98000], [0.75000,   0.28000]]
    J = 0
    D = [[]] * len(theta_weights)
    P = [[]] * len(theta_weights)
    l = 0.0 #Lambda
    activations = [] #Activation of each layer 2-D array
    for i in range(len(train_list)):
        activations = [] #Activation of each layer 2-D array
        print("Processing instance", i+1)
        y = np.array(y_train[i])
        a_input = train_list[i].copy()
        a_input.insert(0,1)
        activations.append(np.array(a_input))
        count = 1
        print("Forward propagating the input", train_list[i])
        print("a", count , a_input)
        count += 1
        for j in range(len(theta_weights)-1):
            z = np.matmul(theta_weights[j], a_input)
            print("z", count, z)
            a_input = g(z)
            a_input.insert(0,1)
            print("a", count, a_input)
            activations.append(np.array(a_input))
            count += 1
        z = np.matmul(theta_weights[-1], a_input)
        print("z", count, z)
        a_input = g(z)
        print("a", count, a_input)
        print("f(x) ", a_input)
        err = np.multiply(-1*y, np.log(a_input)) - np.multiply(1-y, np.log(1-np.array(a_input)))
        err = np.sum(err)
        print("Predicted output for this instance: ", a_input)
        print("Expected output for this instance: ", y_train[i])
        print("Cost, J, associated with this instance: ", err)
        J += err
        print("--------------------------------------------")
        print("Running backpropagation")
        print("Computing gradients based on training instance ", i+1)
        y = y_train[i]
        delta = np.array(a_input) - y
        delta_lst = []
        delta_lst.append(delta)
        print("delta", delta)
        for k in range(len(theta_weights)-1,  0, -1):
            delta = np.multiply(np.multiply(np.matmul(np.transpose(theta_weights[k]), delta), activations[k]) , 1-activations[k])
            delta = np.delete(delta, 0)
            print("delta", delta)
            delta_lst.append(delta)
        delta_lst.reverse()
        delta_lst = np.array(delta_lst, dtype=object)
        for j in range(len(theta_weights)-1,  -1, -1):
            val = np.outer(delta_lst[j], activations[j])
            print(j, "val", val)
            print("\n")
            print("Gradients of Theta", j+1, "based on training instance" , i+1)
            print(val)
            if len(D[j]) == 0:
                D[j] = val
            else:
                D[j] = D[j] + val
    P = [[]] * len(theta_weights)
    print("The entire training set has been processes. Computing the average (regularized) gradients:")
    for i in range(len(theta_weights)-1,  -1, -1):
        print("\n")
        P[i] = np.multiply(theta_weights[i], l)
        P[i][:,0] = 0
        D[i] = np.multiply(D[i]+P[i], 1/len(train_list))
        print("Final regularized gradients of Theta", i+1)
        print(D[i])
    for i in range(len(theta_weights)-1, -1, -1):
        theta_weights[i] = theta_weights[i] - np.multiply(D[i], alpha)
    J = J/len(train_list)
    sum_w = 0
    for i in range(len(theta_weights)):
        for j in range(len(theta_weights[i])):
            for k in range(len(theta_weights[i][j])):
                if k != 0:
                    sum_w += theta_weights[i][j][k]*theta_weights[i][j][k]
    S = l*sum_w/(2*len(train_list))
    cost = J + S
    print("\n")
    print("Final (regularized) cost, J, based on the complete training set:", cost)
    print("\n")
    print("\n")

    



def g(z):
    lst = []
    for neuron in z:
        lst.append(1/(1+math.exp(-neuron)))
    return lst
#Uncomment these lines to test the txt files.
# example_func()
# exit(0)

#Load the dataset


#TO LOAD THE WINE DATASET UNCOMMENT THE FOLLOWIING LINES
# file = r"hw3_wine.csv"
# df = pd.read_csv (file, sep='\t')
# df = df.rename(columns={'# class': 'class'})
# dType_lst = []

#TO LOAD THE HOUSE VOTES DATASET UNCOMMENT THE FOLLOWIING LINES
file = r"hw3_house_votes_84.csv"
df = pd.read_csv (file)
dType_lst = ["#handicapped-infants","water-project-cost-sharing","adoption-of-the-budget-resolution","physician-fee-freeze","el-salvador-adi","religious-groups-in-schools","anti-satellite-test-ban","aid-to-nicaraguan-contras","mx-missile","immigration","synfuels-corporation-cutback","education-spending","superfund-right-to-sue","crime","duty-free-exports","export-administration-act-south-africa"]

dataTypeDict = dict(df.dtypes)
dType_lst = []

#One hot Encode the categorical datasets
def encode_col_in_df(data , cols):
    one_hot_encoded_data = pd.get_dummies(data, columns = cols)
    return one_hot_encoded_data

if len(dType_lst) != 0:
    df = encode_col_in_df(df, dType_lst)

if 0 not in df['class'].tolist():
    for index in df.index:
        df.at[index, 'class'] = df.loc[index].at['class'] - 1

#Split based on class label
print("Splitting datasets")
datasets = {}
split_dataset = df.groupby(df.loc[:,'class'])
for groups, data in split_dataset:
    datasets[groups] = data




#Create stratified k-folds
k = 2
kfolds = {}
for i in range(1,k+1):
    kfolds[i] = pd.DataFrame()
    for key in datasets:
        samples = datasets[key].sample(len(split_dataset.get_group(key))//k, replace=False)
        datasets[key] = datasets[key].drop(samples.index)
        kfolds[i] = pd.concat([kfolds[i], samples])

#Step-size alpha
alpha = 0.5
#Lambda value for regularization
l = 0.25
#Change here the number of hidden layers
n_hidden_layers = 4

#Change here the number of neurons, array i.e number of neurons in each hidden layer
n_neurons = [4,8,4,5]

def initialise_weights():
    theta_weights = [] #Weights of each neuron in each layer 3-D array
    for i in range(n_hidden_layers+1):
        theta_weights.append([])
        if i == 0:
            for j in range(n_neurons[i]):
                d = np.random.uniform( -1, 1, len(df.columns))
                d = d*0.1
                 #+1 for bias neuron would be len(df.columns)-1 as we don't include 'class' column
                theta_weights[i].append(d)
        elif i == n_hidden_layers: #Output layer
            for j in range(len(pd.unique(df['class']))):
                d = np.random.uniform( -1, 1,n_neurons[i-1]+1) #+1 for bias Neuron
                d = d*0.1                
                theta_weights[i].append(d)

        else:
            for j in range(n_neurons[i]):
                d = np.random.uniform( -1, 1,n_neurons[i-1]+1) #+1 for bias Neuron
                d = d*0.1
                theta_weights[i].append(d)    
    return theta_weights

# for i in range(len(theta_weights)):
#     print("layer: ", i+1)
#     print("Number of neurons: ", len(theta_weights[i]))
#     for j in range(len(theta_weights[i])):
#         print("Number of weights for this neuron:", len(theta_weights[i][j]))

metrics = {'acc': [],'rec': [],'prec': [],'f': []} #Dictionary for metrics


for key in kfolds:
    print("For k = ", key)
    test_data = kfolds[key]
    train_data = pd.DataFrame()
    for key1 in kfolds:
        if(key != key1):
            train_data = pd.concat([train_data, kfolds[key1]])
    #test data
    x_test = test_data.loc[:, test_data.columns != 'class']
    x_test = x_test.reset_index(drop=True)
    y_test = test_data.loc[:, 'class']
    y_test = y_test.reset_index(drop=True)
    #train data
    x_train = train_data.loc[:, train_data.columns != 'class']
    x_train = x_train.reset_index(drop=True)
    y_train = train_data.loc[:, 'class']
    y_train = y_train.reset_index(drop=True)

    #Normalize train and test data
 
    for dType in dataTypeDict:
        if file == r"hw3_wine.csv":
            if dType != 'class':
                df[dType] = df[dType].astype(float)
                col_vals = x_train[dType]
                max_val = col_vals.max()
                min_val = col_vals.min()
                for index in x_train.index:
                    val = (x_train.loc[index].at[dType]-min_val)/(max_val-min_val)
                    x_train.at[index, dType] = val


                col_vals = x_test[dType]
                max_val = col_vals.max()
                min_val = col_vals.min()
                for index in x_test.index:
                    val = (x_test.loc[index].at[dType]-min_val)/(max_val-min_val)
                    x_test.at[index, dType] = val

    train_list = x_train.values.tolist()
    oldC = 10000
    cost = 0
    J = 0
    theta_weights = initialise_weights()
    iter = 0
    batch = 20
    batch_lst = []
    J_lst = []
    stop = len(train_list)
    for u in range(0, 1000): 
    # while batch <= stop:
        D = [[]] * len(theta_weights)
        P = [[]] * len(theta_weights)
        oldC = cost
        for i in range(len(train_list)):
        #for i in range(batch):
            activations = [] #Activation of each layer 2-D array
            n = len(pd.unique(df['class']))
            y = np.zeros(n)
            y[int(y_train[i])] = 1
            a_input = train_list[i].copy()
            a_input.insert(0,1)
            activations.append(np.array(a_input)) 
            for j in range(len(theta_weights)-1):
                z = np.matmul(theta_weights[j], a_input)
                a_input = g(z)
                a_input.insert(0,1)
                activations.append(np.array(a_input))
            activations = np.array(activations, dtype=object)
            z = np.matmul(theta_weights[-1], a_input)
            a_input = g(z)
            err = np.multiply(-1*y, np.log(a_input)) - np.multiply(1-y, np.log(1-np.array(a_input)))
            err = np.sum(err)
            J += err
            #Start of Back propagation
            delta = np.array(a_input) - y
            delta_lst = []
            delta_lst.append(delta)
            for k in range(len(theta_weights)-1,  0, -1):
                delta = np.multiply(np.multiply(np.matmul(np.transpose(theta_weights[k]), delta), activations[k]) , 1-activations[k])
                delta = np.delete(delta, 0)
                delta_lst.append(delta)
            delta_lst.reverse()
            delta_lst = np.array(delta_lst, dtype=object)
            for m in range(len(theta_weights)-1,  -1, -1):
                val = np.outer(delta_lst[m], activations[m])
                if len(D[m]) == 0:
                    D[m] = val
                else:
                    D[m] = D[m] + val
        for v in range(len(theta_weights)-1,  -1, -1):
            P[v] = np.multiply(theta_weights[v], l)
            P[v][:,0] = 0
            D[v] = np.multiply(D[v]+P[v], 1/len(train_list))
        for s in range(len(theta_weights)-1, -1, -1):
            theta_weights[s] = theta_weights[s] - np.multiply(D[s], alpha)
        
        J = J/len(x_train)
        sum_w = 0
        for i in range(len(theta_weights)):
            for j in range(len(theta_weights[i])):
                for k in range(len(theta_weights[i][j])):
                    if k != 0:
                        sum_w += theta_weights[i][j][k]*theta_weights[i][j][k]
        
        # J = J/batch
        # J_lst.append(J)
        # batch_lst.append(batch)
        # batch += 20

        S = l*sum_w/(2*n)
        cost = J + S
        if iter == 0:
            oldC = cost
            iter +=1 
            continue
    #Testing by forward propagating
    correct = {}
    incorrect = {}
    for val in df.loc[:, 'class'].value_counts().keys():
        correct[val] = 0
        d = {}
        incorrect[val] = d
    test_list = x_test.values.tolist()
    y_list = y_test.values.tolist()
    for i in range(len(test_list)):
        n = len(pd.unique(df['class']))
        y = np.zeros(n)
        y[int(y_list[i])] = 1
        a_input = test_list[i].copy()
        a_input.insert(0,1)
        for j in range(len(theta_weights)-1):
            z = np.matmul(theta_weights[j], a_input)
            a_input = g(z)
            a_input.insert(0,1)
        z = np.matmul(theta_weights[-1], a_input)
        
        a_input = g(z)
        if np.argmax(y) == np.argmax(a_input):
            correct[np.argmax(y)] = correct[np.argmax(y)] + 1
        else:
            if np.argmax(y) in incorrect[y_list[i]]:
                incorrect[y_list[i]][np.argmax(y)] = incorrect[y_list[i]][np.argmax(y)] + 1
            else:
                incorrect[y_list[i]][np.argmax(y)] = 1

    acc = 0
    prec = 0
    rec = 0
    f = 0
    for i in correct:
        acc += correct[i]
        sum_p = correct[i]
        sum_r = correct[i]
        for j in incorrect:
            if j == i:
                for key1 in incorrect[j]:
                    sum_r += incorrect[j][key1]
            else:
                if i in incorrect[j]:
                    sum_p += incorrect[j][i]
        if sum_p != 0:
            prec += correct[i]/sum_p
        if sum_r != 0:
            rec += correct[i]/sum_r
    acc = acc/len(x_test)
    prec = prec/len(correct)
    rec = rec/len(correct)
    f = prec*rec/(prec+rec)
    metrics["acc"].append(acc)
    metrics["f"].append(f)
    metrics["prec"].append(prec)
    metrics["rec"].append(rec)

metrics["acc"] = np.mean(metrics["acc"])
metrics["f"] = np.mean(metrics["f"])
metrics["prec"] = np.mean(metrics["prec"])
metrics["rec"] = np.mean(metrics["rec"])
print(metrics)

# plt.plot(batch_lst, J_lst)
# plt.xlabel("No. of training instances")
# plt.ylabel("cost-J")
# plt.show()

