import numpy as np 
import tensorflow as tf
from tensorflow import keras
from math import e
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error

    # default hyperparameters from Gianluca
    # kp0 =  0.0001
    # kd0 =  0.5
    # eta_p =  10
    # eta_d =  1

# ------------------------------------ Z SCALING FUNCTIONS -------------------------------------

def scale_z(z, p):
    return (1.15-0.35) * (1 / (1 + np.exp(-z*p))) + 0.35

def scale_z2(z, p):
    return (1.15 - 0.35) * (1 / (1 + tf.exp(-z * p))) + 0.35

# ------------------------------------- LAYERS --------------------------------------------------

# Cell for MF-RNN
class MFCell(keras.layers.Layer):
    # recurrent layer of Memristive-Friendly RNN (MF-RNN)

    def __init__(self,  
                 units, dt=2, kd0=0.5, etap=10, etad=1, alpha=1.0,
                 memory_factor = 0.9, 
                 kp0=0.0001,
                 gamma=1.0, p=1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        self.units = units 
        self.state_size = units
        self.memory_factor = memory_factor  # desired memory factor (spectral radius)
        self.activation = activation

        self.dt = dt
        self.kp0 = kp0
        self.kd0 = kd0
        self.etap = etap
        self.etad = etad
        self.alpha = alpha
        self.gamma = gamma
        self.p = p

        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
        #build the weight matrices
        self.recurrent_kernel = self.add_weight(name='recurrent_kernel', shape=(self.units, self.units), initializer='random_normal', trainable=True)
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)  
        self.bias = self.add_weight(name='bias', shape=(self.units,), initializer='zeros', trainable=True)

        self.built = True


    def call(self, inputs, states):
        
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        state_part = tf.matmul(prev_output, self.recurrent_kernel) 

        z = (input_part + state_part + self.bias)
        z = scale_z2(z, self.p)
        q = (self.kp0*e**(self.etap*z*self.alpha) + self.kd0*e**(-self.etad*z*self.alpha)) * prev_output
        r = self.kp0*e**(self.etap*z*self.alpha) - q
        output = self.dt * r + self.gamma*prev_output
        
        return output, [output]
    

# Cell for MF-ESN
class MFReservoirCell(keras.layers.Layer):
    # Reservoir Cell of Memristive-Friendly ESN (MF-ESN)

    def __init__(self,  
                 units, dt=1, kd0=0.5, etap=10, etad=1, alpha=1.0,
                 memory_factor = 0.9, 
                 kp0= 0.0001, 
                 activation = tf.nn.tanh,
                input_scaling = 1.0, bias_scaling = 1.0, gamma = 1.0, p = 1.0,
                 **kwargs):
        
        self.units = units 
        self.state_size = units
        self.memory_factor = memory_factor  # desired memory factor (spectral radius)
        self.activation = activation

        self.dt = dt
        self.kp0 = kp0
        self.kd0 = kd0
        self.etap = etap
        self.etad = etad
        self.alpha = alpha
        self.gamma = gamma
        self.p = p

        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling

        super().__init__(**kwargs)
        
    def build(self, input_shape):

        #build the recurrent weight matrix
        #uses circular law to determine the values of the recurrent weight matrix
        #rif. paper 
        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli. 
        # "Fast spectral radius initialization for recurrent neural networks."
        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
        value  = (self.memory_factor / np.sqrt(self.units)) * (6/np.sqrt(12))
        W = tf.random.uniform(shape = (self.units, self.units), minval = -value,maxval = value)
        self.recurrent_kernel = W   
        #print('input_shape', input_shape)
        #input("Press Enter to continue...")
        #build the input weight matrix
        self.kernel = tf.random.uniform(shape = (input_shape[-1], self.units), minval = -self.input_scaling, maxval = self.input_scaling)
        #initialize the bias 
        self.bias = tf.random.uniform(shape = (self.units,), minval = -self.bias_scaling, maxval = self.bias_scaling)

        self.built = True


    def call(self, inputs, states):
        
        prev_output = states[0]
        #print(prev_output)
        #input("Press Enter to continue...")
        
        input_part = tf.matmul(inputs, self.kernel)
        state_part = tf.matmul(prev_output, self.recurrent_kernel) 

        z = (input_part + state_part + self.bias)
        z = scale_z(z, self.p)
        q = (self.kp0*e**(self.etap*z*self.alpha) + self.kd0*e**(-self.etad*z*self.alpha)) * prev_output
        r = self.kp0*e**(self.etap*z*self.alpha) - q
        output = self.dt * r + self.gamma*prev_output
        
        return output, [output]


# Cell for ESN
class ReservoirCell(keras.layers.Layer):
    #builds a reservoir as a hidden dynamical layer for a recurrent neural network

    def __init__(self, units, 
                 input_scaling = 1.0, bias_scaling = 1.0,
                 spectral_radius =0.99, 
                 leaky = 1, activation = tf.nn.tanh,
                 **kwargs):
        self.units = units 
        self.state_size = units
        self.input_scaling = input_scaling 
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky #leaking rate
        self.activation = activation
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
        #build the recurrent weight matrix
        #uses circular law to determine the values of the recurrent weight matrix
        #rif. paper 
        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli. 
        # "Fast spectral radius initialization for recurrent neural networks."
        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
        value  = (self.spectral_radius / np.sqrt(self.units)) * (6/np.sqrt(12))
        W = tf.random.uniform(shape = (self.units, self.units), minval = -value,maxval = value)
        self.recurrent_kernel = W   
        
        #build the input weight matrix
        self.kernel = tf.random.uniform(shape = (input_shape[-1], self.units), minval = -self.input_scaling, maxval = self.input_scaling)
                         
        #initialize the bias 
        self.bias = tf.random.uniform(shape = (self.units,), minval = -self.bias_scaling, maxval = self.bias_scaling)
        
        self.built = True


    def call(self, inputs, states):

        prev_output = states[0]
        input_part = tf.matmul(inputs, self.kernel)
        
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation!=None:
            output = prev_output * (1-self.leaky) + self.activation(input_part+ self.bias+ state_part) * self.leaky
        else:
            output = prev_output * (1-self.leaky) + (input_part+ self.bias+ state_part) * self.leaky
        
        return output, [output]
    


# ----------------------------------- MODELS -------------------------------------------------


class MF(keras.Model):
    # Implements a Memristive-Friendly ESN (MF-ESN) model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with MFReservoirCell,
    # followed by a trainable readout layer for classification
    
    def __init__(self, units, kp0, kd0, etap, etad, dt, input_scaling = 1.0, bias_scaling = 1.0, memory_factor = 0.9,
                 readout_regularizer = 1.0, alpha = 1.0, gamma=1.0, p=1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = MFReservoirCell(units = units, dt = dt, kp0 = kp0, kd0 = kd0, etap = etap, etad = etad,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          memory_factor = memory_factor, 
                                                          alpha=alpha, gamma=gamma, p=p))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)


class MFregr(keras.Model):
    # Implements a Memristive-Friendly ESN model (MF-ESN) for time-series regression problems
    #
    # The architecture comprises a recurrent layer with MFReservoirCell,
    # followed by a trainable readout layer for regression
    
    def __init__(self, units, kp0, kd0, etap, etad, dt, input_scaling = 1.0, bias_scaling = 1.0, memory_factor = 0.9,
                 readout_regularizer = 1.0, alpha = 1.0, gamma=1.0, p=1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = MFReservoirCell(units = units, dt = dt, kp0 = kp0, kd0 = kd0, etap = etap, etad = etad,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          memory_factor = memory_factor, 
                                                          alpha=alpha, gamma=gamma, p=p))
        ])
        self.readout = Ridge(alpha = readout_regularizer, solver = 'svd')

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)

        
    def evaluate(self, x, y):
        x_val_states = self.reservoir(x)
        y_pred = self.readout.predict(x_val_states)
        rmse = root_mean_squared_error(y, y_pred)
        return rmse


class ESN(keras.Model):
    #Implements an Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for classification
    
    def __init__(self, units,
                 input_scaling = 1.0, bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = ReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        #print('x_train_states', x_train_states)
        
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)

  
class ESNregr(keras.Model):
    # Implements an Echo State Network model for time-series regression problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for regression
    
    def __init__(self, units,
                 input_scaling=1.0, bias_scaling=1.0, spectral_radius=0.9,
                 leaky=1, 
                 readout_regularizer=1.0,
                 activation=tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
            keras.layers.RNN(cell=ReservoirCell(units=units,
                                                input_scaling=input_scaling,
                                                bias_scaling=bias_scaling,
                                                spectral_radius=spectral_radius,
                                                leaky=leaky))
        ])
        self.readout = Ridge(alpha=readout_regularizer, solver='svd')

    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output

    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)

        
    def evaluate(self, x, y):
        x_val_states = self.reservoir(x)
        y_pred = self.readout.predict(x_val_states)
        
        #rmse = mean_squared_error(y, y_pred, squared=False)
        rmse = root_mean_squared_error(y, y_pred)
        return rmse





class MFcv(keras.Model):
    # Implements a Memristive-Friendly ESN model (MF-ESN) for time-series classification problems
    #
    # The architecture comprises a recurrent layer with MFReservoirCell,
    # followed by a trainable readout layer for classification
    
    def __init__(self, units, kp0, kd0, etap, etad, dt, input_scaling = 1.0, bias_scaling = 1.0, memory_factor = 0.9,
                 readout_regularizer = 1.0, alpha = 1.0, gamma=1.0, p=1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = MFReservoirCell(units = units, dt = dt, kp0 = kp0, kd0 = kd0, etap = etap, etad = etad,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          memory_factor = memory_factor, 
                                                          alpha=alpha, gamma=gamma, p=p))
        ])
        self.readout = RidgeClassifierCV()

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)


class MFregrcv(keras.Model):
    # Implements a Memristive-Friendly ESN model (MF-ESN) for time-series regression problems
    #
    # The architecture comprises a recurrent layer with MFReservoirCell,
    # followed by a trainable readout layer for regression
    
    def __init__(self, units, kp0, kd0, etap, etad, dt, input_scaling = 1.0, bias_scaling = 1.0, memory_factor = 0.9,
                 readout_regularizer = 1.0, alpha = 1.0, gamma=1.0, p=1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = MFReservoirCell(units = units, dt = dt, kp0 = kp0, kd0 = kd0, etap = etap, etad = etad,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          memory_factor = memory_factor, 
                                                          alpha=alpha, gamma=gamma, p=p))
        ])
        #self.readout = Ridge(alpha = readout_regularizer, solver = 'svd')
        self.readout = RidgeCV()

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)

        
    def evaluate(self, x, y):
        x_val_states = self.reservoir(x)
        y_pred = self.readout.predict(x_val_states)
        rmse = root_mean_squared_error(y, y_pred)
        return rmse


class ESNcv(keras.Model):
    #Implements an Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for classification
    
    def __init__(self, units,
                 input_scaling = 1.0, bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 readout_regularizer = 1.0,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = ReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = RidgeClassifierCV()

        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output
    
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        #print('x_train_states', x_train_states)
        
        self.readout.fit(x_train_states, y)
        
    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states,y)

    
class ESNregrcv(keras.Model):
    # Implements an Echo State Network model for time-series regression problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for regression
    
    def __init__(self, units,
                 input_scaling=1.0, bias_scaling=1.0, spectral_radius=0.9,
                 leaky=1, 
                 readout_regularizer=1.0,
                 activation=tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
            keras.layers.RNN(cell=ReservoirCell(units=units,
                                                input_scaling=input_scaling,
                                                bias_scaling=bias_scaling,
                                                spectral_radius=spectral_radius,
                                                leaky=leaky))
        ])
        self.readout = RidgeCV()

    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output

    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)

        
    def evaluate(self, x, y):
        x_val_states = self.reservoir(x)
        y_pred = self.readout.predict(x_val_states)
        
        #rmse = mean_squared_error(y, y_pred, squared=False)
        rmse = root_mean_squared_error(y, y_pred)
        return rmse





