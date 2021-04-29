import os,sys,time,datetime,copy
from ReadData import SQL_CURSOR,GetUpcomingEarnings,AddInfo,ALPHA_TIMESERIES,ConfigTable,ALPHA_FundamentalData,GetTimeSlot
import pandas  as pd
import numpy as np
import base as b
import numpy as np
from techindicators import techindicators

# Tensorflow and Keras
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.utils import class_weight
import keras.backend as K
#from custom_loss import *

# Scipy
from scipy import stats
import numpy as np
import numpy.lib.recfunctions as recfn
import seaborn as sns

training_name='stockEarningsModel'
ReDownload = False
readType='full'
debug=False
draw=True
doPDFs = False
doPlot = False
outdir = b.outdir
import matplotlib.pyplot as plt
import matplotlib
if not draw:
    matplotlib.use('Agg')

def MakePlot(xaxis, yaxis, xname='Date',yname='Beta',saveName='', hlines=[],title='',doSupport=False,my_stock_info=None):
    # plotting
    plt.clf()
    plt.scatter(xaxis,yaxis)
    plt.gcf().autofmt_xdate()
    plt.ylabel(yname)
    plt.xlabel(xname)
    if title!="":
        plt.title(title, fontsize=30)
    for h in hlines:
        plt.axhline(y=h[0],color=h[1],linestyle=h[2]) #xmin=h[1], xmax=h[2],
    if doSupport:
        techindicators.supportLevels(my_stock_info)
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'%s.pdf' %(saveName))
    plt.savefig(outdir+'%s.png' %(saveName))
    if not draw: plt.close()
    plt.close()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([-2000, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  if draw: plt.show()
    
def LoadData():

    connectionCalv2 = SQL_CURSOR('earningsCalendarForTraining.db')
    earningsInfoSaved = pd.read_sql('SELECT * FROM earningsInfo', connectionCalv2)

    return earningsInfoSaved


earningsInfo = LoadData()

print(earningsInfo)

print(len(earningsInfo))

for c in earningsInfo.columns:
    print(c)

# Define decisions for buying or selling
#earningsInfo['label'] = int(0)
#earningsInfo.loc[earningsInfo['twoday_future_return']>3.0e-2,['label']] = 1
#earningsInfo.loc[earningsInfo['oneday_future_return']>5.0e-2,['label']] = 2
#earningsInfo.loc[earningsInfo['oneday_future_return']>10.0e-2,['label']] = 3
#earningsInfo.loc[earningsInfo['oneday_future_return']>15.0e-2,['label']] = 4
#earningsInfo.loc[earningsInfo['oneday_future_return']>20.0e-2,['label']] = 5
#earningsInfo.loc[earningsInfo['twoday_future_return']<-3.0e-2,['label']] = -1
#earningsInfo.loc[earningsInfo['oneday_future_return']<-5.0e-2,['label']] = -2
#earningsInfo.loc[earningsInfo['oneday_future_return']<-10.0e-2,['label']] = -3
#earningsInfo.loc[earningsInfo['oneday_future_return']<-15.0e-2,['label']] = -4
#earningsInfo.loc[earningsInfo['oneday_future_return']<-20.0e-2,['label']] = -5
earningsInfo['label'] = int(2)
earningsInfo.loc[earningsInfo['oneday_future_return']>3.0e-2,['label']] = 3
earningsInfo.loc[earningsInfo['oneday_future_return']>10.0e-2,['label']] = 4
earningsInfo.loc[earningsInfo['oneday_future_return']<-3.0e-2,['label']] = 1
earningsInfo.loc[earningsInfo['oneday_future_return']<-10.0e-2,['label']] = 0

a = tf.keras.utils.to_categorical(earningsInfo['label'], num_classes=5)
print(a)
labelNames=['label']
labelNames = ['label1','label2','label3','label4','label5']
a = pd.DataFrame(a,columns=labelNames)
for c in labelNames:
    earningsInfo[c] = a[c]
#print(earningsInfo['label5'])

#Y_test = np_utils.to_categorical(y_test_encoded)
# preparing data
for c in ['sma50_daybefore','sma20_daybefore','sma200_daybefore']:
    earningsInfo[c+'r']= earningsInfo.adj_close_daybefore/earningsInfo[c]
for c in ['fiveday_prior_vix','thrday_prior_vix','twoday_prior_vix']:
    earningsInfo[c+'r']=earningsInfo[c]/ earningsInfo.adj_close_daybefore

#sys.exit(0)
COLS  = ['sma50_daybeforer','sma20_daybeforer','sma200_daybeforer',
'copp_daybefore',
'daily_return_stddev14_daybefore',
'beta_daybefore',
'alpha_daybefore',
'rsquare_daybefore',
'sharpe_daybefore',
'cci_daybefore',
'stochK_daybefore',
'stochD_daybefore',
'willr_daybefore',
'fiveday_prior_vixr',
'thrday_prior_vixr',
'twoday_prior_vixr',
'earningDiff',
'e_over_p_diff',
'e_over_p_test',
'corr14_daybefore',
#'obv_daybefore',
#'force_daybefore',
#'macd_daybefore',
             ]
earningsInfo.replace([np.inf, -np.inf], np.nan, inplace=True)
earningsInfoRed = earningsInfo[COLS+labelNames+['oneday_future_return']].dropna()
for c in COLS:
    print('%s %s' %(c,earningsInfoRed[c].max()))
print(len(earningsInfoRed))
###############################################################################
# Concatenate and shuffle data
##############################
#np.random.shuffle(earningsInfo) # shuffle data

###############################################################################
# Split train/test data
#######################

data_train, data_test, label_train, label_test = train_test_split(earningsInfoRed, earningsInfoRed[labelNames], test_size=0.2, random_state=0) # 80%/20% train/test split
X_weight = data_train['oneday_future_return'].abs()
X_train = data_train[COLS] # use only COLS
X_test = data_test[COLS] # use only COLS
y_train = label_train
y_test = label_test
print(X_train)
print(y_train.max())

#sns.pairplot(X_train[['sma50_daybeforer','sma20_daybeforer','sma200_daybeforer']], diag_kind='kde')
#if draw: plt.show()
###############################################################################

###############################################################################
# Preprocess data
#################

# Make scaler for train data
scaler = preprocessing.RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75)).fit(X_train) # scaler to standardize data
X_train = scaler.transform(X_train) # apply to train data
X_test = scaler.transform(X_test) # apply to test data

# Save it
#from sklearn.externals import joblib
scaler_filename = "scaler"+training_name+".save"
#joblib.dump(scaler, scaler_filename) 

###############################################################################

###############################################################################
# Weigh classes
################

# This helps address the imbalanced nature of the data
#class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

###############################################################################

###############################################################################
# Define the classifier model
#############################

model = Sequential()
model.add(Dense(32,  activation='relu', input_dim=len(COLS))) #kernel_initializer='normal',
model.add(Dense(16, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1, activation='softmax'))
#model.add(Dense(1))
model.add(Dense(len(labelNames),activation='softmax'))
#model.compile(optimizer='nadam',
#              loss=focal_loss,
#              metrics=['accuracy'])
#model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy']) # best
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) # best
#    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),loss='mean_absolute_error',metrics=['accuracy']) # best
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
#sys.exit(0)
###############################################################################

###############################################################################
# Train the classifier model
############################

# Fit the model
#model.predict(y_train)
history = model.fit(X_train, y_train,  epochs=50, batch_size=50,validation_split = 0.2, sample_weight=X_weight) #, class_weight=class_weight)#, sample_weight=w_train) #validation_split=0.1,
#print(history.history)
plot_loss(history)

# Save the model
model_filename = 'model'+training_name+'.hf'
model.save(model_filename)

###############################################################################

###############################################################################
# Performance Plots
###################
#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#le.fit([0, 1, 2, 3,4,5])
y_pred = np.argmax(model.predict(X_test), axis = 1)
## quick metrics
#y_pred = pd.DataFrame(model.predict(np.array(X_test)),columns=labelNames)
#y_pred_bool = np.argmax(y_pred, axis=1)
#y_pred_bool = np.where(y_pred > 0.3, 1, 0)
y_pred_bool = y_pred
y_test_bool = np.argmax(np.array(y_test[labelNames]), axis = 1)
print(y_pred_bool)
print(y_test_bool)
#y_test_bool = np.where(y_pred > 0.3, 1, 0)
print(classification_report(y_test_bool, y_pred_bool))
print(confusion_matrix(y_test_bool, y_pred_bool))

# calculate the fpr and tpr for all thresholds of the classification
y_pred = pd.DataFrame(model.predict(X_test),columns=labelNames)
print(y_pred)
print(y_test)
y_test = pd.DataFrame(y_test,columns=labelNames)
fpr, tpr, threshold = metrics.roc_curve(y_test[labelNames], y_pred[labelNames],pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

# plot ROC curve
plt.figure()
plt.fill_between(fpr, 0, tpr, facecolor='b', alpha=0.3, label='AUC = {:0.3f}'.format(roc_auc), zorder=0)
plt.plot([0, 1], [0, 1], c='gray', lw=1, ls='--', zorder=1)
plt.plot(fpr, tpr, c='b', lw=2, ls='-', label='ROC Curve', zorder=2)
plt.legend(loc='upper left')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.savefig('ROC'+training_name+'.pdf', bbox_inches='tight')
plt.close()

# plot Prediction versus result
MakePlot( y_test['label'], y_pred['label'], xname='Results',yname='Predictions',saveName='predictedResponse', hlines=[],title='predictedResponse')

## plot PR curve
#average_precision = average_precision_score(y_test, y_pred)
#print('Average precision-recall score: {0:0.2f}'.format(average_precision))
#
#precision, recall, _ = precision_recall_curve(y_test, y_pred)
#
#plt.figure()
#plt.fill_between(recall, precision, alpha=0.2, color='b')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
#plt.savefig('PR'+training_name+'.pdf', bbox_inches='tight', rasterized=False)
#plt.close()

# score predictions
score_train = np.concatenate(model.predict(np.array(X_train)))
score_test = np.concatenate(model.predict(np.array(X_test)))
print('KS test for score_train/score_test: {}'.format(stats.ks_2samp(score_train, score_test)))

# define sig/bkg regions
sig_train = (y_train == 1)
sig_test = (y_test == 1)
bkg_train = (y_train == 0)
bkg_test = (y_test == 0)



# make histograms
## compute KS for sig/bkg prob
#ks_signal = stats.ks_2samp(signal_train, signal_test)[1]
#ks_background = stats.ks_2samp(background_train, background_test)[1]
#print('ks_signal: %s' %ks_signal)
#print('ks_background: %s' %ks_background)
#
## plot output distribution
#plt.figure()
#plt.bar((edges[1:]+edges[:-1])/2, background_train_hist, align='center', width=width, edgecolor=None, facecolor='r', alpha=0.3, label='Background (train)', zorder=1)
#plt.errorbar((edges[1:]+edges[:-1])/2, background_test_hist, yerr=background_test_std, xerr=(edges[1:]-edges[:-1])/2, ecolor='r', elinewidth=1, fmt='none', label='Background (test)', zorder=2)
#plt.bar((edges[1:]+edges[:-1])/2, signal_train_hist, align='center', width=width, edgecolor=None, facecolor='b', alpha=0.3, label='Signal (train)', zorder=1)
#plt.errorbar((edges[1:]+edges[:-1])/2, signal_test_hist, yerr=signal_test_std, xerr=(edges[1:]-edges[:-1])/2, ecolor='b', elinewidth=1, fmt='none', label='Signal (test)', zorder=4)
#
#plt.text(1-0.025, 0.825, 'KS sig (bkg) prob: {:0.3f} ({:0.3f})'.format(ks_signal, ks_background), transform=plt.gca().transAxes, horizontalalignment='right', verticalalignment='top')
#plt.xlim(0, 1)
#plt.ylim(bottom=0)
##plt.grid(zorder=0)
#plt.legend(ncol=2, loc='upper right')
#plt.xlabel('Keras ANN Score')
#plt.ylabel('Events (Normalized)')
#plt.title('Classifier Overtraining Check')
#plt.savefig('overtrain'+training_name+'.pdf', bbox_inches='tight', rasterized=False)
#plt.close()
