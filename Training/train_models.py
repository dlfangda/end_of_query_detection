#!python

#################################INFO####################################
#Functions:
#main functions:
#train_and_pkl_LogisticRegression(episode_path,model_path,\
#                             start=2,end=8,\
#                             modes=['prosodic','lm','prolm'])
#trains the LR models with different featuresets
#saves the trained LR models for different featuresets as pickled files
#
#baseline(episodes_path, threshold=.76)
#writes length of silences to file
#analyse the amount of correct MTP and TRP and returns them into a dict
#
#other functions:
#lsdir(rootdir,ending)
#-> returns a pathlist of files with a specified ending in a directory
#open_pkl_ep(speaker,ep,ep_path)
#-> open one pickled episode and returns it as pandas dataframe
#get_train(speaker,eps_path)
#-> returns the training set (pandas dataframe) with one speaker heldout
#
#used in function baseline
#get_sets(speaker,eps_path)
#-> returns dev, test, train sets (pandas dataframes),
#one speaker is heldout in the training set and used as dev and test set (50:50)
#not used in train LR because you do not need the dev and test set in the training
#baseline_dict(episodes_path, prepath, train=False)
#writes length of silences to files per speaker
#########################################################################


import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from copy import deepcopy
from patsy import dmatrices
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, classification_report
import pickle
import cPickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from EOT_features import acoustic_features, lm_features
from training_utils import round_down, lsdir, open_pkl_ep

sys.path.append("../../")
#from hmm_decoder.hmm import rnn_hmm
#from hmm_decoder.generate_trp_file import gen_trans_file

print "pandas version, needs to be 0.18+", pd.__version__
print "sklearn version, needs to be 0.18+", sklearn.__version__

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
#print "numpy", np.__version__
#print "pickle", pickle.__version__ 

#create the training set
#input: speaker, path to directory of pickled episodes
#return: dataframe for train
def get_train(speaker,eps_path,return_individual_fold=False, limit_of_final_trps=0.75):
    
    
    train = pd.DataFrame() if not return_individual_fold else [] #list if folds
    pathlist = []
    
    print 'Get list of paths...'
    for sp in range(2,8):
        if sp == speaker:
            continue
        else:
            #print sp
            pathlist += lsdir(eps_path+'r'+str(int(sp))+'/', '.pkl')

    print 'Load pickled episodes into Dataframe...'
    lsp=1
    speaker_dict = {} # a dict of lists of dataframes (for each episode)
    for path in pathlist:
        ep = path.split('_')[1].split('.pkl')[0]
        speak = path.split('_')[0][1]
        eppath = eps_path+'r'+str(int(speak))+'/'#+path
        if int(speak) != int(lsp):
            print "speaker", speak
            speaker_dict[int(speak)] = []
            #print train
            lsp = speak
        #if lsp =='4':  #just for testing
        #    del speaker_dict[int(speak)]
        #    break
        #print speak,ep,eppath
        features = acoustic_features + " + " +\
                              lm_features[lm_features.find('time_in_sec +')+13:]
        train_ep = open_pkl_ep(speak,ep,eppath)
        #print "first test"
        #y, X = dmatrices(features, train_ep, return_type="dataframe")
        train_ep = train_ep.drop(train_ep[(train_ep.label == 2)&(train_ep.label_dur>limit_of_final_trps)].index)
        #to test the input works for all features
        #print "second test"
        try:
            y, X = dmatrices(features, train_ep, return_type="dataframe")
        except:
            raise Exception("Couldn't load in all lm and acoustic features for {} {} {}".format(speak,ep,eppath))
        #train = pd.concat([train, train_ep])
        if return_individual_fold:
            speaker_dict[int(speak)].append(deepcopy(train_ep)) #adding the episode to the speaker list
        else:            
            train = train.append(deepcopy(train_ep))
    print "training data loaded"
    #print train
    #quit()
    if return_individual_fold:
        #a list of lists of dataframes
        #train = [speaker_dict[key] for key in sorted(speaker_dict.keys())]
        return speaker_dict
    return train


def fold_batches_with_padding(features, speaker_fold, seq_len, scaler, limit_of_final_trps=None):
    #returns a list of tuples of X,y for each batch
    batches = []
    for fold in speaker_fold: #each episode
        if limit_of_final_trps:
            fold = fold.drop(fold[(fold.label == 2.0)&(fold.label_dur>limit_of_final_trps)].index)
        y, X = dmatrices(features, fold, return_type="dataframe")
        if len(X) == 0:
            print "leaving out null episode"
            continue
        X = scaler.transform(X)
        fold_dataX = []
        for i in range(0,X.shape[0]):
            fold_dataX.append(X[i-seq_len:i])
        X = pad_sequences(fold_dataX, maxlen=seq_len, dtype='float32')
        #print "fold with padding new shape", X.shape
        y = np.ravel(y)
        batches.append((X,y))
    return batches

def train_and_save_LSTM(train,features,n_features,n_classes,scaler,filepath,model=None,seq_len=10):
    """Re-formats data for the LSTM, creates an LSTM model if not using a saved one
    
    train :: a list of dataframes for each speaker
    filepath :: where the best model will be stored
    model :: saved model to load"""
    #first, format the data into one big list with padding
    #The data must be formatted into a list of matrices to be fed to our network.
    print 'Formatting Data'
    batchSize = 1  #how many samples before updating
    feature_len = n_features


    print 'Building model...'
    #set return_sequences=False for the last hidden layer.
    if not model:
        #FROM http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
        model = Sequential()
        
        #simple net
        model.add(LSTM(68, batch_input_shape=(10, seq_len, feature_len),return_sequences=False, stateful=True))
        
        #more complex net
        #model.add(LSTM(68, stateful=True,  return_sequences=True,
        #          input_shape=(seq_len, feature_len)))
        #model.add(Dropout(0.3))
        #model.add(LSTM(38, stateful=True, activation='tanh',
        #               input_length=seq_len))

        #final layer and loss always the same
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics=['fmeasure'])
        
    #model.reset_states()
    print 'Training...' 
    num_epochs = 10
    best = 0
    best_epoch = 0
    best_model = None
    for e in range(num_epochs):
        print "epoch", e+1
        scores = []
        #TODO val function and stopping criterion
        #hold out 10% in each fold and get loss on that
        for fold in train:
            batches = fold_batches_with_padding(features, fold, seq_len, scaler,limit_of_final_trps=0.75)
            split_point = int(0.9 * len(batches))
            batch_train, batch_test = batches[:split_point], batches[split_point:]
            print "train/test split", len(batch_train), len(batch_test)
            for X,y in batch_train:
                X = np.reshape(X, (X.shape[0], seq_len, feature_len))
                y = np_utils.to_categorical(y.astype(int)) #convert to 1-hot output
                if not y.shape[1] == n_classes:
                    print "bad y shape"
                    continue
                totalTimeSteps = len(X) # Each element in the data represents one timestep of our single feature.
                X = X[:round_down(totalTimeSteps, 10)] # for batches need to round down
                y = y[:round_down(totalTimeSteps, 10)]
                #print 'Formatted X of shape', X.shape
                #print "Formatted y of shape", y.shape
                #print y
                #mycallback = ModelCheckpoint(filepath, monitor='val_fmeasure', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
                #print X, y
                h = model.fit(X, y, nb_epoch=1, batch_size=10, verbose=0, shuffle=False)
                model.reset_states() # in stateful this is needed
            for X,y in batch_test:
                X = np.reshape(X, (X.shape[0], seq_len, feature_len))
                totalTimeSteps = len(X)
                X = X[:round_down(totalTimeSteps, 10)] # for batches need to round down
                y = y[:round_down(totalTimeSteps, 10)]
                #if not y.shape[1] == n_classes:
                #    print "bad y shape"
                #s    continue
                #y = np_utils.to_categorical(y.astype(int)) #convert to 1-hot output
                predictions = model.predict_classes(X,batch_size=10, verbose=1)
                p_r_f_tags = precision_recall_fscore_support(y,predictions, average='weighted')
                test_result = p_r_f_tags[2]
                scores.append(test_result)
                model.reset_states() # in stateful this is needed
            print "rolling val score", np.average(scores)
        #scores = model.evaluate(X, y, batch_size=batchSize, verbose=0) #just test on last one
        #model.reset_states()
        score = np.average(scores)*100
        print "Model Accuracy: %.2f%%" % score, "after epoch", e+1
        if score > best:
            best = score
            best_epoch = e
            best_model = model
            model.save(filepath)
            print "new best", best, "on epoch", e+1
            continue
        if e - best_epoch >= 6: break # no improvement after 6
    print 'training complete'
    return best_model 

def scaler_for_all_folds(train, features):
    print "Training scaler..."
    scaler = StandardScaler() #scaler for data
    print "for", len(train), "folds"
    for speaker_fold in train:
        print len(speaker_fold)
        for fold in speaker_fold:
            y, X = dmatrices(features, fold, return_type="dataframe")
            if len(X) == 0:
                print "leaving out null episode"
                continue
            scaler.fit(X)
            n_features = X.shape[1]
    print "scaler", n_features
    return scaler, n_features

##################################
#loading train is time consuming
#get_sets(speaker=2,eps_path='./pickledEpisodes/')
###################################

#loop over all speaker and training logistic regression models
#with each speaker as heldout data
#input: path to directories with pickled episodes
#and were to save the pickled models
#optional: start at speaker, end at speaker and list of modes
#return: nothing
def train_and_pkl_model(model_name, episodes_path, models_path,\
                             start=2, end=8,\
                             modes=['acoustic','lm','acousticlm']):


    #label_dur  -> using this feature is cheating ;-)
    ######################################################
    acoustic_and_lmfeatures = acoustic_features + " + " +\
                        lm_features[lm_features.find("wml "):]
    
    print 'get training data for all speakers...'
    train_all = get_train(0,episodes_path,return_individual_fold='lstm'in model_name, limit_of_final_trps=0.75)
    if 'lstm' in model_name:
        print "dict of list of dataframes with keys", train_all.keys()
    else:
        print "one big training data frame of shape", train_all.shape
    
    for speaker in range(start,end):
        print speaker
        #you don't need the dev and test set for training purposes
        if "lstm" in model_name:
            train = [ train_all[k] for k in train_all.keys() if k != speaker ]
        else:
            train = train_all[train_all['speaker']!=float(speaker)]
        for mode in modes:
            print 'training LR model with mode ',mode,\
                  'and heldout speaker ',str(speaker)
            #first, get the relevant features to test on
            if mode == 'acoustic':
                features = acoustic_features
            elif mode == 'lm':
                features = lm_features
            elif mode == 'acousticlm':
                features = acoustic_and_lmfeatures
            
            
            #model_name = "Logistic_Regression"
            #model_name = "lstm"
            print 'using', model_name, 'model for training...'
            modpath = models_path + model_name
            if not os.path.isdir(modpath):
                os.makedirs(modpath)

            prepath = modpath+'/r'+str(speaker)
            if not os.path.isdir(prepath):
                os.makedirs(prepath)
                
            if "lstm" in model_name:
                print 'prepare data...'
                scaler, n_features = scaler_for_all_folds(train, features)
                n_classes = 3
                print 'building and training model...'
                model = train_and_save_LSTM(train, features, n_features, n_classes, scaler, prepath+'/'+str(speaker)+'_'+mode+r'.h5')                
            elif 'Logistic_Regression' in model_name:
                print 'prepare data...'
                y, X = dmatrices(features, train, return_type="dataframe")
                print "new shape X", X.shape
                print "new shape y", y.shape

                print 'ravel...'
                y = np.ravel(y)

                print 'building and training model...'
                scaler = StandardScaler() #scaler for data
                X = scaler.fit_transform(X)
                
                #bring in class_weights as:
                len_all = float(len(train))
                weight_trp = len(train[train['label']==2])/len_all
                weight_mtp = len(train[train['label']==0])/len_all
                weight_s = len(train[train['label']==1])/len_all

                model = LogisticRegression(solver='lbfgs', class_weight={0:weight_mtp, 1:weight_s, 2:weight_trp}, multi_class='multinomial')
                model.fit(X, y)

            print "write scaler to file..."
            spath = prepath+"/"+str(speaker)+'_'+mode+r'_scaler.pkl'
            with open(spath,'wb') as fid:
                cPickle.dump(scaler, fid)

            print 'write model to file...'
            mpath = prepath+'/'+str(speaker)+'_'+mode
            if "lstm" in model_name:
                model.save(mpath+r'.h5')
            else:
                joblib.dump(model,mpath +r'.pkl')

            #print 'compute accuracies for speaker '+str(speaker)+' with mode '+mode
            #train_acc = model.score(X, y)
            
            #print 'train accuracy ',train_acc
            

        train = None
      
        y = None
        X = None
      
    return


if __name__ == "__main__":
    episodes_path = THIS_DIR + "/../../../eot_detection_data/Data/pickled_episodes/"
    models_path = THIS_DIR +  "/../../../eot_detection_data/Models/"
    model_name = 'lstm_5'#'lstm' #or LogisticRegression, MMT
    ##########################################################
    #here you can set the values for the start and end speaker,
    #the modes you want to produce a pickled model for
    #and the treshold for the baseline
    #the default values are set to create all models and the baseline threshold at .75
    #that means the EOTaccuracy for a model that cuts-in after 750ms of silence
    ##########################################################
    #start = 2
    #end = 8
    #modes = ['acoustic','lm','acousticlm']
    #threshold = .75
    
    train_and_pkl_model(model_name, \
                        episodes_path, \
                        models_path,\
                        start=2,\
                        end=8,\
                        modes=['acoustic','lm','acousticlm'])
    #print the baseline for a threshold
    #base_dict = baseline(episodes_path, models_path, threshold=.76)
