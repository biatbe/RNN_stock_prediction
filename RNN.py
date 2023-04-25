import numpy as np
import math
import yahoofinancials
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class RNN:

    def __init__(self):
        # Load stock data/rearrange if necessary
        self.df = yf.download('TSLA')
        self.df = self.df[['Adj Close','Open', 'Volume', 'High', 'Low']].round(2)
        self.x_train, self.y_train, self.x_test, self.scaler = self.preprocess(self.df)
        self.input_size = 5
        self.hidden_size = 256
        self.output_size = 1

        self.Wxh = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Why = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bh = np.zeros((1, self.hidden_size))
        self.by = np.zeros((1, self.output_size))

    def preprocess(self, df, time_steps=5, period=1):

        # Splitting into train and test data
        train_size = int(len(df) * 0.80)
        train_data = df[:train_size].iloc[:,0:1].values
        test_data = df[train_size:].iloc[:,0:1].values
        train_len = len(train_data)
        test_len = len(test_data)

        # Scaling the data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(train_data)
        # It's also possible to only scale some part of the data
        # scaled_data = scaler.fit_transform(train_data[["Open", "Adj Close", "Volume"]])

        X_train = []
        y_train = []
        for i in range(time_steps, train_len - 1):
            X_train.append(scaled_data[i - time_steps:i, 0])
            y_train.append(scaled_data[i:i + period, 0])
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Reshaping X_train for efficient modelling
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        inputs = pd.concat((df[:train_len].iloc[:,0:1], df[train_len:].iloc[:,0:1]), axis=0).values
        inputs = inputs[len(inputs) - (test_len + time_steps):]
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        # Preparing X_test
        X_test = []
        for i in range(time_steps, test_len):
          X_test.append(inputs[i - time_steps:i, 0])
        print(X_test)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, y_train, X_test, scaler

    def forward_pass(self, x, hprev):
        # Hidden layer output
        h = np.tanh(np.dot(hprev, self.Whh) + np.dot(x, self.Wxh) + self.bh)
        # Output
        y = np.dot(h, self.Why) + self.by
        # Store values needed for backward pass
        cache = (x, h, hprev)
        return y, h, cache

    def backward_pass(self, dy, dhnext, cache):
        # Values from cache
        x, h, hprev = cache
        # Calculate gradients
        dWhy = np.dot(h.T, dy)
        dby = dy
        dh = np.dot(dy, self.Why.T) + dhnext
        dhraw = (1 - h ** 2) * dh
        dbh = np.sum(dhraw, axis=0, keepdims=True)
        dWxh = np.dot(x.T, dhraw)
        dWhh = np.dot(hprev.T, dhraw)
        dx = np.dot(dhraw, self.Whh.T)
        # Return gradients and hidden state gradients for next timestep
        dhprev = np.dot(dhraw, self.Whh.T)
        return dx, dhprev, dWxh, dWhh, dWhy, dbh, dby

    def train(self, X, Y, num_epochs=10, learning_rate=0.1):
        # Initialize hidden state
        hprev = np.zeros((1, self.hidden_size))
        # Initialize best_loss to a high value
        best_loss = [1000.0, 1000.0]
        values = None
        # Loop over epochs
        for epoch in range(num_epochs):
            # Initialize loss
            loss = 0
            # Loop over timesteps
            for t in range(len(X)):
                # Forward pass
                x = X[t].reshape((1, self.input_size))
                y_true = Y[t]
                y_pred, h, cache = self.forward_pass(x, hprev)
                # Compute loss
                loss += (y_pred - y_true) ** 2
                # Backward pass
                dy = 2 * (y_pred - y_true)
                dx, dhprev, dWxh, dWhh, dWhy, dbh, dby = self.backward_pass(dy, np.zeros_like(hprev), cache)
                # Update weights and biases
                self.Wxh -= learning_rate * dWxh
                self.Whh -= learning_rate * dWhh
                self.Why -= learning_rate * dWhy
                self.bh -= learning_rate * dbh
                self.by -= learning_rate * dby
                # Update hidden state for next timestep
                hprev = h

            # Check for new best_loss
            if int(sum(loss[0])*10000) < int(sum(best_loss)*10000):
                best_loss = loss[0]
                values = (self.Wxh, self.Whh, self.Why, self.bh, self.by)
            # Print loss every 10 epochs
            if epoch % 10 == 0:
                print('Epoch', epoch, 'Loss', loss, 'Best Loss', best_loss)
        return values

    def predict(self, x, Wxh, Whh, Why, bh, by):
        hprev = np.zeros((1, self.hidden_size))
        # Hidden layer output
        h = np.tanh(np.dot(hprev, Whh) + np.dot(x, Wxh) + bh)
        # Output
        y = np.dot(h, Why) + by
        return y

    def runRnn(self, data=None):
        values = self.train(self.x_train, self.y_train)
        preds = []
        for i in range(len(self.x_test)):
            pred = self.predict(self.x_test[i].reshape((1, self.input_size)), values[0], values[1], values[2], values[3], values[4])
            preds.append(self.scaler.inverse_transform(pred)[0])
        return preds

    def actual_pred_plot(self, preds):
        x = self.df[len(self.df) - len(preds):].index.get_level_values("Date")
        y1 = self.df[len(self.df) - len(preds):].iloc[:, 0:1].values
        y2 = np.concatenate(preds).ravel()
        plt.plot(x, y1, label="actual")
        plt.plot(x, y2, label="preds")
        plt.legend()
        plt.show()
