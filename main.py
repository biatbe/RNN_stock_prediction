from RNN import RNN

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rnn = RNN()
    pred = rnn.runRnn()
    print(pred)
    rnn.actual_pred_plot(pred)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
