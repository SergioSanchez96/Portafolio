import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def suma(a,b):
    print(a+b)

def load_timeseries(ric, file_extension='csv'):
    path_file = 'D://EDUCACION//Finanzas//Finanzas_Cuantitativas_Py//data//'+ric+'.'+file_extension
    tabla_bruta = pd.read_csv(path_file)
    tabla = pd.DataFrame()
    tabla['date'] = pd.to_datetime(tabla_bruta['Date'],dayfirst=True)
    tabla['close'] = tabla_bruta['Close'].values
    tabla.sort_values(by='date', ascending=True)    # para verificar orden correcto
    tabla['close_previous'] = tabla_bruta['Close'].shift(1) # deslizas los valores de Close hacia abajo
    tabla['return_close'] = tabla['close']/tabla['close_previous']-1
    tabla = tabla.dropna()
    tabla = tabla.reset_index(drop=True)
    x = tabla['return_close']
    x_str = 'Retorno reales '+ric
    return x, x_str, tabla  # serie tiempo activo individual

def plot_timeseries(tabla, ric):
    plt.figure()
    plt.plot(tabla['date'], tabla['close'])
    plt.title('Serie de tiempo precios real de '+ ric)
    plt.xlabel('Años')
    plt.ylabel('Precio')
    plt.show()

def plot_histogram(x, x_str,plot_str):
    plt.figure()
    plt.hist(x,bins=100)
    plt.title('Histograma' + x_str)
    plt.xlabel(plot_str)
    plt.show()

def sincronizar_seriestiempo(ric, benchmark):
    # cargar datos
    x1, str1, tabla1 = load_timeseries(ric)
    x2, str2, tabla2 = load_timeseries(benchmark)

    # sincronizar series de tiempo
    timestamp1 = list(tabla1['date'].values)
    timestamp2 = list(tabla2['date'].values)
    timestamps = list(set(timestamp1) & set(timestamp2))

    tabla1_sync = tabla1[tabla1['date'].isin(timestamps)]
    tabla1_sync.sort_values(by='date', ascending=True)
    tabla1_sync = tabla1_sync.reset_index(drop=True)

    tabla2_sync = tabla2[tabla2['date'].isin(timestamps)]
    tabla2_sync.sort_values(by='date', ascending=True)
    tabla2_sync = tabla2_sync.reset_index(drop=True)

    t = pd.DataFrame()          # serie tiempo 2 activos en conjunto
    t['date'] = tabla1_sync['date']
    t['price_1'] = tabla1_sync['close']
    t['price_2'] = tabla2_sync['close']
    t['return_1'] = tabla1_sync['return_close']
    t['return_2'] = tabla2_sync['return_close']

    y = t['return_1'].values        # retornos de activo
    x = t['return_2'].values        # retornos de mercado
    
    return x, y, t