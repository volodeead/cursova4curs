import streamlit as st
import pandas as pd
from datetime import timedelta
from prophet.serialize import model_to_json, model_from_json
from prophet import Prophet
import plotly_express as px

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import tensorflow as tf

#Title
st.title("Передбачення курсів валют")

typeModel = st.selectbox("Виберіть модель: ", {"LSTM", "Prophet"})
typeVallet = st.selectbox("Виберіть валюту: ", {"£", "€", "¥", "AUD"})

def getTimeSeriesData(A, window=7):
    X, y = list(), list()
    for i in range(len(A)):
        end_ix = i + window
        if end_ix > len(A) - 1:
            break
        seq_x, seq_y = A[i:end_ix], A[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def interactive_plot_Proph(df):
    plot = px.scatter(df, x=df['ds'], y=df['yhat'])
    st.plotly_chart(plot)

def interactive_plot_LSTM(df):
    plot = px.scatter(df, x=df['Date'], y=df['Predicted'])
    st.plotly_chart(plot)

def prophet_pred(df_target, date_size):
    data0 = pd.read_csv("Foreign_Exchange_Rates.csv")
    cols = data0[data0[df_target]=='ND'].index.tolist()
    data0=data0.drop(index=cols)

    print(len(data0))

    target = df_target

    target = target.replace(' ', '')
    target = target.replace('/', '')
    target = target.replace('$', '')
    target = target.replace('-', '')

    with open('Prophet/'+ target +'.json', 'r') as fin:
        model = model_from_json(fin.read())
    df_for_test = model.make_future_dataframe(periods=date_size)
    df_for_test = df_for_test[df_for_test['ds'].notna()]
    test_forecast = model.predict(df_for_test)
    figure = model.plot(test_forecast)
    st.write(figure)
    train_data_model = pd.DataFrame({'Predicted': test_forecast['yhat'],
                            'Date': test_forecast['ds']})
    st.write(train_data_model.tail(timePeriod))
    interactive_plot_Proph(test_forecast)

def lstm_pred(df_target, date_size):
    import pandas as pd 
    import numpy as np 
    df = pd.read_csv("Foreign_Exchange_Rates.csv")
    df = df.drop(columns=["Unnamed: 0"])
    newColumnsNames = list(map(lambda c: c.split(" - ")[0] if "-" in c else "DATE", df.columns))
    df.columns = newColumnsNames
    df = df.replace("ND", np.nan)
    df = df.bfill().ffill() 
    df = df.set_index("DATE")
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    num_records = 7303
    data = {}
    data["DATE"] = pd.date_range("2000-01-03", "2019-12-31", freq="D")
    complete = pd.DataFrame(data=data)
    complete = complete.set_index("DATE")
    complete = complete.merge(df, left_index=True, right_index=True, how="left")
    complete = complete.bfill().ffill()
    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense
    toInspect = ["JAPAN", "UNITED KINGDOM", "EURO AREA", "AUSTRALIA"]

    df_target_column = df_target

    if(df_target == 'UNITED_KINGDOM'):
        df_target_column = "UNITED KINGDOM"

    if(df_target == 'EURO_AREA'):
        df_target_column = "EURO AREA"

    sampled2d = complete.resample("2D").mean()
    window = 30
    num_features = 1
    X, y = getTimeSeriesData(list(sampled2d[df_target_column]), window=window)
    print("X:", X.shape)
    print("Y:", y.shape)
    X = X.reshape((X.shape[0], X.shape[1], num_features))
    print("-----------")
    print("X:", X.shape)
    print("Y:", y.shape)
    x_train = X[:2900]
    x_test = X[2900:]
    y_train = y[:2900]
    y_test = y[2900:]
    new_model = tf.keras.models.load_model('LSTM/'+ df_target +'_model.h5')
    yPred1111 = new_model.predict(x_test, verbose=0)
    yPred1111.shape = yPred1111.shape[0]


    fig = plt.figure(figsize=(20, 8))
    sns.set(rc={"lines.linewidth": 1})
    sns.lineplot(x=np.arange(y_train.shape[0]), y=y_train, color="green")
    sns.set(rc={"lines.linewidth": 1})
    sns.lineplot(x=np.arange(yPred1111.shape[0]) + 2899, y=yPred1111, color="coral")
    sns.set(rc={"lines.linewidth": 1})
    sns.lineplot(x=np.arange(y.shape[0]), y=y, color="gray", alpha = 0.5)
    plt.legend(["Train", "Test", "All Data"])

    st.text("Зображення зміни ціни, сірим і оранжевим позначені значення")
    st.text("на яких тестувалася модель")
    st.pyplot(fig)


    new_array_x = []

    iterator = 0
    y_pred_in_fut = 0

    for i in sampled2d[df_target_column]:
        new_array_x.insert(len(new_array_x),i)
        iterator = iterator + 1

    for date in range(date_size):
        x_for_pred = X[-1].reshape(1, 30, 1)

        new_x = new_model.predict(x_for_pred, verbose=0)

        new_x.shape = new_x.shape[0]

        last = len(new_array_x)
        new_array_x.insert(last, new_x[0])
        X, y_pred_in_fut = getTimeSeriesData(new_array_x, window=window)

    train_data_model = pd.DataFrame({'Predicted': y_pred_in_fut,
                                     'Date': np.arange(y_pred_in_fut.shape[0])})
    st.write(train_data_model.tail(timePeriod))

    interactive_plot_LSTM(train_data_model)





if (typeModel == "Prophet"):
    timePeriod = st.selectbox("Дальність прогнозу(дні): ", {1, 3, 5, 10, 30, 60, 360})
    if st.button('Зробити прогноз моделлю Prophet'):
        if (typeVallet == "AUD"):
            prophet_pred('AUSTRALIA - AUSTRALIAN DOLLAR/US$', timePeriod)
        if (typeVallet == "£"):
            prophet_pred('UNITED KINGDOM - UNITED KINGDOM POUND/US$', timePeriod)
        if (typeVallet == "€"):
            prophet_pred('EURO AREA - EURO/US$', timePeriod)
        if (typeVallet == "¥"):
            prophet_pred('JAPAN - YEN/US$', timePeriod)

if (typeModel == "LSTM"):
    timePeriod = st.selectbox("Дальність прогнозу(дні): ", {3, 5, 10})
    if st.button('Зробити прогноз моделлю LSTM'):
        if (typeVallet == "AUD"):
            lstm_pred('AUSTRALIA', timePeriod)
        if (typeVallet == "£"):
            lstm_pred('UNITED_KINGDOM', timePeriod)
        if (typeVallet == "€"):
            lstm_pred('EURO_AREA', timePeriod)
        if (typeVallet == "¥"):
            lstm_pred('JAPAN', timePeriod)


