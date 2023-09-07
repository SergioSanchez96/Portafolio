import numpy as np
import pandas as pd
#import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew,kurtosis, chi2, linregress

import Funciones
importlib.reload(Funciones) # necesito las funciones establecidas en la pestaña Funciones, con esto actualizo cambios sin problemas

class metricas():

    # buenas practicas inicialziar todas la variables que serán usadas 
    # ademas necesitamos las variales para plot_str
    def __init__(self,ric):
        self.ric = ric
        self.returns = []
        self.size = 0
        self.str_name = None
        self.mean = 0.0
        self.std = 0.0
        self.skew = 0.0
        self.kurtosis = 0.0 # excess kurtosis
        self.median = 0.0
        self.quartile_25 = 0.0
        self.var_95 = 0.0
        self.cvar_95 = 0.0
        self.sharpe = 0.0
        self.jarque_bera = 0.0
        self.p_value = 0.0 # equivalently jb < 6
        self.is_normal = 0.0
        # self.percentile_25 = None
        # self.percentile_75 = None

    # para mostarr print(variable de clase Metrica): indica como string el resumem de metricas
    def __str__(self): 
        str_self = self.str_name +'| size: '+ str(self.size)+'\n'+ self.plot_str()
        return str_self
    
    def load_timeseries(self):
        self.returns, self.str_name, self.tabla = Funciones.load_timeseries(self.ric) # serie tiempo activo individual
        self.size = self.tabla.shape[0]
        # self.size = len(self.returns)
    
    def generate_random_vector(self, type_random_variable, size=10**6, degrees_freedom=None):
        self.size = size
        if type_random_variable == 'normal':
            self.returns = np.random.standard_normal(size)
            self.str_name = 'Standard Normal RV'
        elif type_random_variable == 'exponential':
            self.returns = np.random.standard_exponential(size)
            self.str_name = 'Exponential Normal RV'
        elif type_random_variable == 'student':
            if degrees_freedom == None:
                degrees_freedom = 750
            self.returns = np.random.standard_t(df=degrees_freedom, size=size)
            self.str_name = 'Student Rv (df = ' + str(degrees_freedom)  + ')'
        elif type_random_variable == 'chi-squared':
            if degrees_freedom == None:
                degrees_freedom = 750
            self.returns = np.random.chisquare(df=degrees_freedom, size=size)
            self.str_name = 'Chi-squared Rv (df = ' + str(degrees_freedom)  + ')'
        

    def compute(self):
        self.size = self.tabla.shape[0]
        self.mean = np.mean(self.returns)
        self.std = np.std(self.returns)
        self.skew = skew(self.returns)
        self.kurtosis = kurtosis(self.returns) # kurtosis en exceso
        self.sharpe = self.mean/self.std*np.sqrt(252)
        self.median = np.median(self.returns)
        self.quartile_25 = np.percentile(self.returns, 25)
        self.var_95 = np.percentile(self.returns,5)
        self.cvar_95 =  np.mean(self.returns[self.returns<= self.var_95])
        self.jarque_bera = self.size/6*(self.skew**2 + 1/4*self.kurtosis**2) # equivalently jb < 6
        self.p_value = 1 - chi2.cdf(self.jarque_bera, df=2)
        self.is_normal = (self.p_value >= 0.05) #  si cumple es normal

    def plot_str(self):
        round_digits = 4
        plot_str = 'mean ' + str(np.round(self.mean,round_digits))\
            + ' | std dev ' + str(np.round(self.std,round_digits))\
            + ' | skewness ' + str(np.round(self.skew,round_digits))\
            + ' | kurtosis ' + str(np.round(self.kurtosis,round_digits)) + '\n'\
            + 'Jarque Bera ' + str(np.round(self.jarque_bera,round_digits))\
            + ' | p-value ' + str(np.round(self.p_value,round_digits))\
            + ' | is normal ' + str(self.is_normal) + '\n'\
            + 'Sharpe annual ' + str(np.round(self.sharpe,round_digits))\
            + ' | VaR 95% ' + str(np.round(self.var_95,round_digits))\
            + ' | CVaR 95% ' + str(np.round(self.cvar_95,round_digits))
        return plot_str
    
    def plot_timeseries(self):
        Funciones.plot_time_series_price(self.tabla, self.ric)

    def plot_histogram(self):
        Funciones.plot_histogram(self.returns, self.str_name, self.plot_str())
    

class capm_manager():

    def __init__(self, ric, benchmark, round_digits=4):
        self.round_digits = round_digits
        self.ric = ric
        self.benchmark = benchmark
        # introducir todas las variables con las que trabajaras mas adelante en la clase
        self.x = []     # retornos de mercado
        self.y = []     # retornos de activo
        self.t = pd.DataFrame()

    def __str__(self):
        str_plot = 'Linear regression | ric: '+ self.ric\
                + ' | benchmark: ' + self.benchmark + '\n'\
                + ' alpha(intercept): ' + str(self.alpha)\
                + ' | beta(slope): ' + str(self.beta) + '\n'\
                + 'p-value: ' + str(self.p_value)\
                + ' | null hypothesis: ' + str(self.null_hypothesis) + '\n'\
                + 'r-value: '+str(self.r_value)\
                + ' | r-squared: '+str(self.r_squared)
        return str_plot

    def load_timeseries(self):
        self.x, self.y , self.t = Funciones.sincronizar_seriestiempo(self.ric, self.benchmark)

    def compute(self):
        # regresion lineal de ric respecto a benchmark
        slope, intercept, r_value, p_value, std_err = linregress(self.x,self.y)
        self.beta = np.round(slope, self.round_digits)       # beta
        self.alpha = np.round(intercept, self.round_digits) # alfa
        self.p_value = np.round(p_value, self.round_digits)
        #H0 = beta e intercepccion = 0 (no hay regresion lineal)
        # H0: 
        self.null_hypothesis = (p_value > 0.05) # p_value < 0.05 -> rechazar hipotesis Nula. 
        self.r_value = np.round(r_value, self.round_digits) # coeficiente de correlacion
        self.r_squared = np.round(r_value**2, self.round_digits) # % de info que recuperamos de la regresion
        self.predictor_linreg = self.alpha + self.beta*self.x      # beta*rend_mercado + intercepcion (alfa)

    def scatterplot(self):
        str_title = 'Scatterplot of returns' + '\n' + self.__str__()
        plt.figure()
        plt.title(str_title)
        plt.scatter(self.x,self.y)
        plt.plot(self.x, self.predictor_linreg, color ='green') # X= MERCADO, Y = ACTIVO
        plt.ylabel(self.ric)
        plt.xlabel(self.benchmark)
        plt.grid()
        plt.show()

    def plot_normalised(self):
        # normalizados a 100
        price_ric = self.t['price_1']
        price_benchmark = self.t['price_2']
        plt.figure(figsize=(12,5))
        plt.title('Serie de tiempo de precios | normalizados a 100')
        plt.xlabel('Periodos')
        plt.ylabel('Precios Normalizados')
        price_ric = 100 * price_ric/price_ric[0]
        price_benchmark = 100 * price_benchmark/price_benchmark[0]
        plt.plot(self.t['date'], price_ric, color='blue', label=self.ric)
        plt.plot(self.t['date'], price_benchmark, color='red', label= self.benchmark)
        plt.legend(loc=0)
        plt.grid()
        plt.show()

    def plot_dual_axes(self):
        plt.figure(figsize=(12,5))
        plt.title('Serie de tiempo de precios')
        plt.xlabel('Periodos')
        plt.ylabel('Precios')
        ax = plt.gca()
        ax1 = self.t.plot( kind='line', x='date',y='price_1', ax=ax ,color='blue', grid=True, label = self.ric)
        ax2 = self.t.plot(kind='line', x='date', y='price_2', ax=ax, color='red', grid=True, secondary_y=True, label=self.benchmark)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()

