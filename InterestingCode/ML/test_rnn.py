# univariate stacked lstm example
from numpy import array
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
#from keras.utils import plot_model
#from keras.utils.vis_utils import plot_model
import tensorflow as tf
import sys,os,datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
MAX_EPOCHS = 20

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=None, val_df=None, test_df=None,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)
    
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    
    return inputs, labels
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)
  
    ds = ds.map(self.split_window)
  
    return ds

  def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} [normed]')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
               label='Inputs', marker='.', zorder=-10)
    
      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index
    
      if label_col_index is None:
        continue
    
      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)
    
      if n == 0:
        plt.legend()
    
    plt.xlabel('Time [h]')
    plt.show()
  @property
  def train(self):
    return self.make_dataset(self.train_df)
  
  @property
  def val(self):
    return self.make_dataset(self.val_df)
  
  @property
  def test(self):
    return self.make_dataset(self.test_df)
  
  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result

#WindowGenerator.make_dataset = make_dataset
#WindowGenerator.train = train
#WindowGenerator.val = val
#WindowGenerator.test = test
#WindowGenerator.example = example
        
# split a univariate sequence
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def LSTMModel(n_steps, n_features):
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu',return_sequences=True)) #, input_shape=(n_steps, n_features)))
   # LSTM(32, return_sequences=True)
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

  
def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
  print(window.train)
  print(window.val)
  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


# read in data
df = pd.read_csv('climate/jena_climate_2009_2016.csv')
column_indices = {name: i for i, name in enumerate(df.columns)}
print(df)
# slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
print(df)

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
print('split samples')
sys.stdout.flush()
num_features = df.shape[1]
print('done shaping')
sys.stdout.flush()
train_mean = train_df.mean()
train_std = train_df.std()
print('starting normalized')
sys.stdout.flush()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
print(train_df)
print('create windows')
sys.stdout.flush()
val_performance = {}
performance = {}
doTrain=False
# Define the windows
w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                         train_df=train_df, val_df=val_df, test_df=test_df,
                     label_columns=['T (degC)'])
print('w2 windows')
print(w2)
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
     train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['T (degC)'])
print(single_step_window)

# convolutional
CONV_WIDTH = 3
conv_window = WindowGenerator(input_width=CONV_WIDTH,
    label_width=1,
    shift=1,train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['T (degC)'])
wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['T (degC)'])
print(wide_window)
for example_inputs, example_labels in wide_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1, train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['T (degC)'])

# Start the training
if doTrain:
    # start with the baseline
    baseline = Baseline(label_index=column_indices['T (degC)'])

    baseline.compile(loss=tf.losses.MeanSquaredError(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

    IPython.display.clear_output()
    val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
    print(performance['Baseline'])

# linear
print('linear')
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
if doTrain:
    history = compile_and_fit(linear, single_step_window)

    val_performance['Linear'] = linear.evaluate(single_step_window.val)
    performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)
    wide_window.plot(linear)

# dense
print('dense')
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
if doTrain:
    history = compile_and_fit(dense, single_step_window)

    val_performance['Dense'] = dense.evaluate(single_step_window.val)
    performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

    wide_window.plot(dense)

print('dense multistep')
multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])
if doTrain:
    history = compile_and_fit(multi_step_dense, conv_window)

    IPython.display.clear_output()
    val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
    performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)
    conv_window.plot(multi_step_dense)

#conv_model = tf.keras.Sequential([
#    tf.keras.layers.Conv1D(filters=32,
#                           kernel_size=(CONV_WIDTH,),
#                           activation='relu'),
#    tf.keras.layers.Dense(units=32, activation='relu'),
#    tf.keras.layers.Dense(units=1),
#])
#history = compile_and_fit(conv_model, conv_window)
#
#IPython.display.clear_output()
#val_performance['Conv'] = conv_model.evaluate(conv_window.val)
#performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

print('LSTM')
print(np.version.version)
print(tf.__version__)
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)])
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)

history = compile_and_fit(lstm_model, wide_window)
IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
wide_window.plot(lstm_model)


if False:
# define input sequence
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    print(X, y)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = LSTMModel(n_steps,n_features)

    # setup window
    wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['T (degC)'])

    
    history = compile_and_fit(model, wide_window)

    #IPython.display.clear_output()
    #val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
    #performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)

    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    
    # demonstrate prediction
    x_input = array([70, 80, 90])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)
