import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import yfinance as yf
import datetime as dt 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM 



# Method 1: Create data from multiple companies
def create_data(companies, start, end, scaler):
    data = {}
    for company in companies:
        data[company] = yf.download(company, start=start, end=end)

    scaled_data = scaler.fit_transform(data[companies[0]]['Close'].values.reshape(-1, 1))

    prediction_days = 60
    x_train = []
    y_train = []

    for company in companies:
        company_data = scaled_data  # You may want to adjust this depending on how you want to use multiple companies' data

        for x in range(prediction_days, len(company_data)):
            x_train.append(company_data[x - prediction_days:x, 0])
            y_train.append(company_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data to fit the LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, data

# Method 2 : Create and Train Model
def create_and_train_model(x_train, y_train, epochs=50, batch_size=64):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction value
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Method 3: Create and Train Model
def model_accuracy(company, start, end, data, model, scaler):
    prediction_days = 60
    test_data = yf.download(company, start=start, end=end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data[company]['Close'], test_data['Close']))

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)

    # Fit the scaler on the training data
    scaler.fit(model_inputs)
    model_inputs = scaler.transform(model_inputs)

    # Test Prediction
    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return actual_prices, predicted_prices, model_inputs


# Method 4 : Plotting prediction for a company
def plot_pred(actual_prices, predicted_prices) :
    plt.plot(actual_prices, color ="black", label =f"Actual {company} Price")
    plt.plot(predicted_prices, color ="red", label =f"Predicted {company} Price")
    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel('Share Price')
    plt.legend()
    return plt.show()

# Method 5 : Prediction Next Day
def next_day_pred(model, model_inputs, scaler):
    prediction_days = 60

    scaler.fit(model_inputs)
    model_inputs = scaler.transform(model_inputs)

    real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    return print(f"Prediction : {prediction[0,0]}")


# Main
companies=['GE','GOOG','AAPL']
train_start = dt.datetime(2000,1,1)
train_end = dt.datetime(2020,1,1)

scaler = MinMaxScaler(feature_range=(0, 1))

x_train, y_train, data = create_data(companies, train_start, train_end, scaler)

model = create_and_train_model(x_train, y_train)

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

for company in companies : 
    actual_prices, predicted_prices, model_inputs = model_accuracy(company, test_start, test_end, data, model, scaler)
    plot_pred(actual_prices, predicted_prices)
    next_day_pred(model, model_inputs, scaler)