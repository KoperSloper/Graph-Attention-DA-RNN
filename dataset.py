import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from financetoolkit import Toolkit
from api import api_key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_date="2010-01-01"

companies = ['AAPL', 'MSFT', 'GOOG', 'TSM', 'V', 'LVMUY', 'AMZN', 'NVDA','ADBE', 'NFLX', 'WMT', 'MA',
             'KO', 'NKE','INTC', 'IBM', 'QCOM', 'CSCO', 'TXN', 'EBAY', 'NDAQ', 'SMCI', 'JPM', 'HD', 'PEP',
             'COST', 'F', 'BA', 'BAC', 'XOM', 'CVX', 'PG', 'VZ', 'JNJ', 'DIS', 'GE', 'PFE', 'MRK', 'UNH',
             'CAT', 'MMM', 'HON', 'DD', 'AXP', 'MCD', 'TRV', 'GS', 'PRU', 'ALL', 'BMY', 'BSX', 'C', 'CL', 
             'ADSK', 'CHKP', 'AMAT', 'CTSH', 'EXPE', 'FAST', 'GILD', 'HSIC', 'IDXX', 'INCY']

lookback = 20

def get_data(companies: list, api_key: str, start_date: str):
    historical_data = Toolkit(tickers = companies, api_key = api_key, start_date = start_date).get_historical_data()
    historical_data = historical_data.fillna(method='ffill')

    data_stacked = historical_data.stack().reset_index()

    data_stacked = data_stacked.fillna(0)

    data = data_stacked.rename(columns={'level_1': 'Stock'})
    data = data.drop('Dividends', axis=1)

    data_pivoted = data.pivot(index='Date', columns='Stock')
    data_pivoted.columns = data_pivoted.columns.swaplevel(0, 1)
    data_pivoted = data_pivoted.sort_index(axis=1)

    data_np = data_pivoted.to_numpy().reshape(len(data_pivoted), len(data_pivoted.columns.levels[0]), len(data_pivoted.columns.levels[1]))

    return data_np

data_np = get_data(companies, api_key, start_date)

def split_data(data, lookback):
    X_list = []
    y_list = []
    target_list = []

    for index in range(len(data) - lookback):
        X_list.append(data[index: index + lookback, 1:, :])
        y_list.append(data[index: index + lookback, 0, :])

        target_list.append(data[index + lookback, 0, 0])

    X = np.array(X_list)
    y = np.array(y_list)
    target = np.array(target_list)

    return [X, y, target]

X, y, target = split_data(data_np, lookback)

def get_dataloaders(X, y, target):
    train_length = int(len(X) * 0.7)
    val_length = int(len(X) * 0.15)
    test_length = len(X) - train_length - val_length

    X_train = X[:train_length]
    y_train = y[:train_length]
    target_train = target[:train_length]

    X_val = X[train_length:train_length + val_length]
    y_val = y[train_length:train_length + val_length]
    target_val = target[train_length:train_length + val_length]

    X_test = X[train_length + val_length:]
    y_test = y[train_length + val_length:]
    target_test = target[train_length + val_length:]

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    X_train_2D = X_train.reshape(-1, X_train.shape[-1])
    scaler_X.fit(X_train_2D)

    y_train_2D = y_train.reshape(-1, y_train.shape[-1])
    scaler_y.fit(y_train_2D)

    target_train_2D = target_train.reshape(-1, 1)
    scaler_target.fit(target_train_2D)

    X_train_scaled = scaler_X.transform(X_train_2D).reshape(X_train.shape)
    y_train_scaled = scaler_y.transform(y_train_2D).reshape(y_train.shape)
    target_train_scaled = scaler_target.transform(target_train_2D).flatten()

    # Transform the validation data
    X_val_2D = X_val.reshape(-1, X_val.shape[-1])
    X_val_scaled = scaler_X.transform(X_val_2D).reshape(X_val.shape)

    y_val_2D = y_val.reshape(-1, y_val.shape[-1])
    y_val_scaled = scaler_y.transform(y_val_2D).reshape(y_val.shape)

    target_val_2D = target_val.reshape(-1, 1)
    target_val_scaled = scaler_target.transform(target_val_2D).flatten()

    # Transform the test data
    X_test_2D = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler_X.transform(X_test_2D).reshape(X_test.shape)

    y_test_2D = y_test.reshape(-1, y_test.shape[-1])
    y_test_scaled = scaler_y.transform(y_test_2D).reshape(y_test.shape)

    target_test_2D = target_test.reshape(-1, 1)
    target_test_scaled = scaler_target.transform(target_test_2D).flatten()

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    target_train_tensor = torch.tensor(target_train_scaled, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    target_val_tensor = torch.tensor(target_val_scaled, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    target_test_tensor = torch.tensor(target_test_scaled, dtype=torch.float32)

    train_dataset=TensorDataset(X_train_tensor,y_train_tensor,target_train_tensor)
    val_dataset=TensorDataset(X_val_tensor,y_val_tensor,target_val_tensor)
    test_dataset=TensorDataset(X_test_tensor,y_test_tensor,target_test_tensor)

    train_loader=DataLoader(train_dataset,batch_size=128,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=128,shuffle=False,drop_last=True)
    test_loader=DataLoader(test_dataset,batch_size=128,shuffle=False,drop_last=True)

    return (train_loader,val_loader,test_loader, scaler_target)

train_loader, val_loader, test_loader, scaler_target = get_dataloaders(X, y, target)

def get_edge_index():
    edges = []
    for i in range(len(companies)-1):
        for j in range(i + 1, len(companies)-1):
            edges.append([i, j])
            edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)
    return edge_index

edge_index = get_edge_index()