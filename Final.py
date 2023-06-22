import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import sklearn.neural_network as nn
import sklearn.model_selection as ms


data = pd.read_csv("A:/ML/CarCo2/CO2 Emissions_Canada.csv")


# *********************charecter to number*******************

def str2num(row):
    a = 1
    vehicleClass = {}
    for i in range(len(data[str(row)])-1):
        if data[str(row)][i] == data[str(row)][i+1]:
            # vehicleClass={data[str(row)][i]:a}
            vehicleClass[data[str(row)][i]] = a
        else:
            # vehicleClass={data[str(row)][i]:a}
            vehicleClass[data[str(row)][i]] = a
            a = a+1

    data[str(row)] = [vehicleClass[item] for item in data[str(row)]]


str2num('Make')
str2num('Model')
str2num('Transmission')
str2num('Fuel Type')
str2num('Vehicle Class')


data = data.to_numpy()

inputs = data[:, :11]
outputs = data[:, 11]

# *********************spiliting & normalizing data*******************

trainx, testx, trainy, testy = ms.train_test_split(
    inputs, outputs, train_size=0.7, random_state=1, shuffle=True)

scaler = pp.MinMaxScaler(feature_range=(0, 1))

trainx = scaler.fit_transform(trainx)
testx = scaler.fit_transform(testx)

trainy = scaler.fit_transform(trainy.reshape(-1, 1))
testy = scaler.fit_transform(testy.reshape(-1, 1))

# *********************learning*******************

MLP = nn.MLPRegressor(hidden_layer_sizes=(55, 12), activation='relu',
                      solver='adam', max_iter=400, alpha=0.001)
MLP.fit(trainx, trainy)

# *********************score*******************

train_acc = MLP.score(trainx, trainy)
test_acc = MLP.score(testx, testy)

print('tracc: ', train_acc)
print('teacc: ', test_acc)
