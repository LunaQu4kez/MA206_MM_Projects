import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import datetime

data = pd.read_excel(r"C:\Users\charl\Desktop\数学建模\Project\Project 4\附件1(Attachment 1)2023-51MCM-Problem B.xlsx").values

lines = np.unique(np.array([i[0]+i[1]for i in data[:,[1,2]]]))
lines = np.array([[i[0],i[1]] for i in lines])

all_pre = []

for i in range(lines.shape[0]):
    y = data[np.logical_and(data[:,1]==lines[i,0],data[:,2]==lines[i,1])][:,-1].astype(np.float64)
    # 将数据划分为训练集和测试集
    train_size = int(len(y) * 0.8)
    train, test = y[:train_size], y[train_size:]

    # 数据归一化
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train.reshape(-1, 1))
    test = scaler.transform(test.reshape(-1, 1))

    # 创建数据生成器
    def create_dataset(data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back), 0])
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 7
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    # 重塑数据以适应LSTM模型的输入格式
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 创建LSTM模型
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)

    # 预测
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # 将预测结果转换回原始尺度
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 计算预测准确性，例如使用均方误差（MSE）作为评估指标
    train_mse = np.mean((train_predict - y_train) ** 2)
    test_mse = np.mean((test_predict - y_test) ** 2)

    # 计算预测日期与最后一个训练日期之间的天数
    last_train_date = datetime.date(2019, 4, 17)
    start_pred_date = datetime.date(2019, 4, 18)
    end_pred_date = datetime.date(2019, 4, 20)
    days_to_predict = (end_pred_date-start_pred_date).days

    # 使用训练数据的最后一部分来开始预测
    input_data = train[-look_back:]

    predictions = []

    # 预测每一天的货量
    for i in range(days_to_predict):
        input_data_reshaped = input_data.reshape(1, look_back, 1)
        pred = model.predict(input_data_reshaped)
        predictions.append(pred[0, 0])

        # 更新输入数据，用预测值替换最早的值
        input_data = np.roll(input_data, -1)
        input_data[-1] = pred

    # 将预测值转换回原始尺度
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    tdays = []
    # 打印预测结果
    for i, pred in enumerate(predictions, start=1):
        pred_date = start_pred_date + datetime.timedelta(days=i - 1)
        tdays += [pred[0]]
    all_pre+=[tdays]

all_pre = np.array(all_pre)
all_pre
#pd.DataFrame(np.c_["1",lines,all_pre],columns=["起点","终点","2023年4月28日","2023年4月29日"]).to_excel(r"C:\Users\charl\Desktop\数学建模\Project\Project 4\Problem 2.xlsx")
