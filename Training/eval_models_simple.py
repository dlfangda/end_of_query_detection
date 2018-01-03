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
#sys.path.append("../../../simple_rnn_disf/rnn_disf_detection/") #path to disf. detection repo
#from rnn_disf_detection.rnn.elman import Elman
#from rnn_disf_detection.rnn.lstm import Lstm
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from train_models import get_train,  scaler_for_all_folds,fold_batches_with_padding
from EOT_features import acoustic_features, lm_features
from training_utils import round_down, lsdir, open_pkl_ep

sys.path.append("../../")
#from hmm_decoder.hmm import rnn_hmm
#from hmm_decoder.generate_trp_file import gen_trans_file

print "pandas version, needs to be 0.18+", pd.__version__
print "sklearn version, needs to be 0.18+", sklearn.__version__
#print "numpy", np.__version__
#print "pickle", pickle.__version__ 



#create the training set
#input: speaker, path to directory of pickled episodes
#return: dataframe for train
def get_test(speaker,eps_path,return_individual_fold=False, limit_of_final_trps=0.75):
    
    train = pd.DataFrame() if not return_individual_fold else [] #list if folds
    pathlist = []
    
    print 'Get list of paths...'
    for sp in range(2,8):
        if sp != speaker:
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
            
        train_ep = open_pkl_ep(speak,ep,eppath)
        train_ep = train_ep.drop(train_ep[(train_ep.label == 2)&(train_ep.label_dur>limit_of_final_trps)].index)
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

#create train, dev and test set
#input: speaker, directory path of pickled episodes
#needs
#the file dev_test_eps.csv, to reproduce sets consistently
#the functions above (get train and open pkl ep)
#return: dataframes for train, dev, test set
def get_sets(speaker,eps_path,train=False):

    ep_path = eps_path+'r'+str(int(speaker))+'/'
    
    dev = pd.DataFrame()
    test = pd.DataFrame()
    train = pd.DataFrame()
    
    dev_test_episodes = open('./dev_test_eps.csv','r')
    eps = dev_test_episodes.readlines()
    dev_test_episodes.close()
    
    eps_dict = {}
    for line in eps:
        line = line.strip()
        line = line.split("\t")
        eps_dict[line[0]] = line[1].split(',')
    #print eps_dict
    
    eps_dev =  eps_dict['dev_'+str(float(speaker))]
    eps_test =  eps_dict['test_'+str(float(speaker))]
    #print len(eps_dev)
    #print len(eps_test)

    print 'create dev set with ',len(eps_dev),' episodes...'
    for ep in sorted(eps_dev):
        test_ep = open_pkl_ep(speaker,ep,ep_path)
        test = pd.concat([test, test_ep])
    print 'create test set...',len(eps_test),' episodes...'
    for ep in sorted(eps_test):
        dev_ep = open_pkl_ep(speaker,ep,ep_path)
        dev = pd.concat([dev, dev_ep])

    eps = None
    #speaker_eps = None
    eps_dict = None

    if train:
        print 'create training set with speaker ',str(speaker),' as heldout data...'
        train = get_train(speaker,eps_path)
        return dev, test, train
    else:
        return dev, test


#incremental Viterbi training with probability distributions for label 
def get_ep_dict_incremental(prob_dist_df, t_ep, tagdict, raw_tagdict, hmm):
    num_eps = len(prob_dist_df.episode.unique())
    counter = 0
    print 'number of episodes:',num_eps
    ep_seq_dict = {}
    epgroup = prob_dist_df.groupby('episode')
    for ep, group in epgroup:
        counter+=1
        print 'progress:',(counter/float(num_eps))*100,'percent'
        idx_array = []
        #from the raw input without time
        #if not ep == t_ep:
        #    continue
        for k,v in sorted(tagdict.items(),key=operator.itemgetter(1)):
            #print k
            for raw_tag in raw_tagdict.keys():
                if k.startswith(raw_tag):
                    idx_array.append(deepcopy(group[raw_tag].values))
        #change shape  
        idx_array = np.asarray(idx_array,dtype="float32")
        idx_array = np.swapaxes(idx_array,0,1)

        update_list = []
        
        hmm.viterbi_init()
        for i in range(0,len(idx_array)):
            b_sequence = hmm.viterbi_incremental(idx_array,a_range=(i,i+1),changed_suffix_only=True)
            #print b_sequence
            update_list.append(deepcopy(b_sequence))
        ep_seq_dict[ep] = deepcopy(update_list)
    return ep_seq_dict

#write incremental results from MMT to a file
def write_dict_to_file_incremental(epdict,speaker,incremental='',mode='lm',subset='dev', models_path='./../../'):
    modpath = models_path+'MMT'
    if not os.path.isdir(modpath):
        os.makedirs(modpath)
    
    prepath = modpath+'/r'+str(speaker)
    if not os.path.isdir(prepath):
        os.makedirs(prepath)
        
    dict_file = codecs.open(prepath+'/viterbi'+incremental+str(speaker)+mode+'_'+subset+'.csv','w','utf8')
    for ep in epdict:
        dict_file.write(str(ep))
        dict_file.write('\t')
        ep_list = [';'.join(label) if type(label) == list else label for label in epdict[ep]]
        dict_file.write(','.join(ep_list))
        dict_file.write('\n')
    dict_file.close()
    return

#open probability distribution file for a speaker 
#with mode lm, prosodic or prolm
#and subset dev or test
def train_MMT(mode, subset, LG_path, speak):
    #create tagdict
    tag_df = pd.read_csv(r'./../../hmm_decoder/models/trp.csv',sep='\t')
    #print 'converted labels\n',tag_df.keys()

    raw_tagdict = {'trp':2,'speech':1,'mtp':0}
    tagdict = {key: ind for ind, key in zip([ind for ind in range(0,len(tag_df.keys())-1)],list(tag_df.keys()[1:]))}

    #print '\n tagdict\n',tagdict

    hmm = rnn_hmm(tagdict,markov_model_file=r'./../../hmm_decoder/models/trp.csv')

        
    fname = LG_path+'/r'+str(speak)+'/prob_dist_r'+str(speak)+'_'+mode+'_'+subset+'.csv'
    y = pd.read_csv(fname)
    x = y
    x = x[['episode','time','mtp','speech','trp']]
    t_ep = random.choice(x.episode.unique())
        
    ep_seq_dict = get_ep_dict_incremental(x, t_ep, tagdict, raw_tagdict, hmm)
    write_dict_to_file_incremental(ep_seq_dict,speaker=speak, incremental='incremental', mode=mode, subset=subset)

    return


def test_LSTM(train,features,n_features,n_classes,scaler,filepath,model=None,seq_len=10):
    """Re-formats data for the LSTM, creates an LSTM model if not using a saved one
    
    train :: a list of dataframes for each speaker
    filepath :: where the best model will be stored
    model :: saved model"""
    #first, format the data into one big list with padding
    #The data must be formatted into a list of matrices to be fed to our network.
    print 'Formatting Data'
    batchSize = 1  #how many samples before updating
    feature_len = n_features

    print 'Loading model...'
    #set return_sequences=False for the last hidden layer.
    #if not model:
    #    #FROM http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
    #    model = Sequential()
    #    
        #model.add(LSTM(16, input_shape=(X.shape[1], X.shape[2])))
    #    model.add(LSTM(16, input_shape=(seq_len, feature_len)))
#
     #   #model.add(LSTM(32, input_shape=(X.shape[1], 1)))
    #    model.add(Dense(n_classes, activation='softmax'))
     #   model.compile(loss='categorical_crossentropy', optimizer='adam', 
    #                  metrics=['fmeasure'])
    
    model = load_model(filepath) 
    
    #model.reset_states()
    print 'Testing...' 
    num_epochs = 1
    best = 0
    best_epoch = 0
    best_model = None
    #for e in range(num_epochs):
   #     print "epoch", e+1
    scores = []
    all_labels = []
    all_predictions = []
    for fold in train: #should only be one fold!
        predictions = []
        labels = []
        batches = fold_batches_with_padding(features, fold, seq_len, scaler,limit_of_final_trps=0.75)
        for X,y in batches:
            X = np.reshape(X, (X.shape[0], seq_len, feature_len))
            totalTimeSteps = len(X)
            X = X[:round_down(totalTimeSteps, 10)] # for batches need to round down
            y = y[:round_down(totalTimeSteps, 10)]
            #if not y.shape[1] == n_classes:
            #    print "bad y shape"
            #s    continue
            #print 'Formatted X of shape', X.shape
            #print "Formatted y of shape", y.shape
            #print y
            #mycallback = ModelCheckpoint(filepath, monitor='val_fmeasure', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
            #h = model.fit(X, y, nb_epoch=1, batch_size=50, validation_split=0.1, verbose=2)
            #scores.append(h.history['val_fmeasure'][-1])
            print y
            print y.shape
            #batch_predictions = model.predict_classes(X, batch_size=1, verbose=2)
            batch_predictions = model.predict_classes(X,batch_size=10, verbose=1)
            
            #test_result = model.evaluate(X, y, batch_size=batchSize, verbose=2) #just test on last one
            print "---"
            
            print batch_predictions
            print batch_predictions.shape
            labels.extend(y)
            predictions.extend(batch_predictions)
            model.reset_states() #in stateful reset the states
        
        #test_result = float(tp)/float(len(predictions))
        all_labels.extend(labels)
        all_predictions.extend(predictions)
        p_r_f_tags = precision_recall_fscore_support(labels,predictions, average='weighted')
        test_result = p_r_f_tags[2]
        #print p_r_f_tags
        fold_tag_summary =  classification_report(labels,predictions,labels=
                                     [0,1,2],target_names=['MTP','S','TRP'])
        print fold_tag_summary
        scores.append(test_result)
            
        model.reset_states()
        score = np.average(scores)*100
        print "Model Accuracy: %.2f%%" % score
    
    tag_summary =  classification_report(all_labels,all_predictions,labels=
                                     [0,1,2],target_names=['MTP','S','TRP'])
    print 'testing complete'
    return tag_summary + "\n" + str(score) + "\n%%%%%%%%%%%%%\n"

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
def test_model_labels(model_name, episodes_path, models_path,\
                             start=2, end=8,\
                             modes=['acoustic','lm','acousticlm']):

    acoustic_and_lmfeatures = acoustic_features + " + " +\
                              lm_features[lm_features.find('time_in_sec +')+13:]

    #uncomment all the rows below for first results (accuracy scores)
    #also comment out get_train and uncomment get_sets above
    
    #results_file = open("results_labels_" + model_name + "_allatonce_2_layer_68_38_nodes.text","w")
    results_file = open("results_labels_" + model_name + "_stateful_1_layer_68_nodes" + ".text","w")
    for speaker in range(start,end):
        #you don't need the dev and test set for training purposes
        #dev,test,train = get_sets(speaker,episodes_path)
        #NB here just doing cross val on the training episodes of heldout speaker
        #train = get_train(speaker,episodes_path,return_individual_fold=model_name=='lstm')
        train_all = get_test(speaker,episodes_path,return_individual_fold='lstm' in model_name)
        print len(train_all), "test dict"
        if 'lstm' in model_name:
            test = [ train_all[k] for k in train_all.keys() if k == speaker ]
        else:
            test = train_all[train_all['speaker'] == float(speaker)]
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
            modpath = models_path + model_name #+ "_stateful_1_layer_68_nodes"
            #modpath = models_path+model_name
           # if not os.path.isdir(modpath):
            #    os.makedirs(modpath)

            prepath = modpath+'/r'+str(speaker)
            #if not os.path.isdir(prepath):
            #    os.makedirs(prepath)
                
            load_scaler = True
            spath = prepath+"/"+str(speaker)+'_'+mode+r'_scaler.pkl'
            if 'lstm' in model_name:
                
                if load_scaler: 
                    print "loading scaler..." #if already loaded
                    _, n_features = scaler_for_all_folds(test, features)
                    with open(spath,'rb') as fd:
                        scaler = cPickle.load(fd)
                else:
                    print "training scaler..."
                    scaler, n_features = scaler_for_all_folds(train, features)

                
                n_classes = 3
                print 'testing model...'
                results_string = test_LSTM(test, features, n_features, n_classes, scaler, prepath+'/'+str(speaker)+'_'+mode+r'.h5') 
                results_file.write(str(speaker) + ":" + mode + "\n")
                results_file.write(str(results_string))
            elif'Logistic_Regression' in model_name:
                print 'prepare data...'
                test = test.drop(test[(test.label == 2)&(test.label_dur>0.75)].index)
                y, X = dmatrices(features, test, return_type="dataframe")
                print "new shape X", X.shape
                print "new shape y", y.shape
                #a, B = dmatrices(features, dev, return_type="dataframe")
                #c, D = dmatrices(features, test, return_type="dataframe")
                print 'ravel...'
                y = np.ravel(y)
                #a = np.ravel(a)
                #c = np.ravel(c)
                print "loading model..."
                model = joblib.load(prepath+'/'+str(speaker)+'_'+mode+r'.pkl')
                
                if load_scaler: 
                    print "loading scaler and transforming data..." #if already loaded
                    with open(spath,'rb') as fd:
                        scaler = cPickle.load(fd)
                    X = scaler.transform(X)
                else:
                    "training scaler and tranforming data ..."
                    scaler = StandardScaler() #scaler for data
                    X = scaler.fit_transform(X)
                print 'testing model...'
                predictions = model.predict(X)
                #model = LogisticRegression(solver='lbfgs', class_weight={1:0.14, 0:0.12, 2:0.74}, multi_class='multinomial')
                #model.fit(X, y)
                print predictions
                print predictions.shape
                tp = 0
                #new_y = []
                #for p in range(len(predictions)):
                #    for i in range(0,3):
                ##        if y[p][i] == 1.0:
                 #           new_y.append(i)
                 #           break
                
                #test_result = float(tp)/float(len(predictions))
                p_r_f_tags = precision_recall_fscore_support(y,predictions, average='weighted')
                score = p_r_f_tags[2]
                #print p_r_f_tags
                tag_summary =  classification_report(y,predictions,labels=
                                             [0,1,2],target_names=['MTP','S','TRP'])
                print tag_summary
                results_string = tag_summary + "\n" + str(score) + "\n%%%%%%%%%%%%%\n"
                results_file.write(str(speaker) + ":" + mode + "\n")
                results_file.write(str(results_string))
                
                

            elif model_name =='MMT':
                print 'Viterbi training...'
                LG_path = models_path+'Logistic_regression'
                if not os.path.isdir(LG_path+'/r'+str(speaker)):
                    print 'no probability distributions \
from Logistic regression model for speaker',speaker, ' at\n', LG_path
                    continue
                subset = 'test'#or dev ;)
                train_MMT(mode, subset, LG_path, speaker)
                continue
            
            #NB change this
            if not load_scaler:
                print "writing scaler to file!"
                with open(spath,'wb') as fid:
                    cPickle.dump(scaler, fid)

            #print 'compute accuracies for speaker '+str(speaker)+' with mode '+mode
            #train_acc = model.score(X, y)
            #dev_acc = model.score(B, a)
            #test_acc = model.score(D, c)

            #print 'train accuracy ',train_acc
            #print 'development accuracy ',dev_acc
            #print 'test accuracy ',test_acc

        train = None
        #dev = None
        #test = None

        y = None
        X = None
        #a = None
        #B = None
        #c = None
        #D = None
    results_file.close()
    return

#input: path to the pickled episodes 
#return: dict {speaker:{number_of_eps,silence_lengthes:[]}
def baseline_dict(episodes_path, prepath, train=False):
    
    dur_dict = {}
    
    for sp in range(2,8):
        sp_df = pd.DataFrame()

        print 'create dev and test set for speaker', sp
        dev, test = get_sets(sp, episodes_path, train=False)
        sp_df = pd.concat([dev, test])

        print 'get silence durations...'
        eps_groups = sp_df.groupby('episode')
        dur_dict[sp] = {'ep_num':len(eps_groups),'label_durs':[]}
        for ep,epgroup in eps_groups:
            if int(ep)%15==0:
                print ep
            pruef = False
            epgroup = epgroup[epgroup['time_in_sec']>1.0]
            last_label = ''
            last_dur = ''
            #i = 0
            for i in range(len(epgroup)):
                #row = pd.Series(row)
                #print row
                label = epgroup.iloc[[i]].label.values[0]
                if label == 2:
                    if ep == 2:
                        print 'angekommen'
                    dur_dict[sp]['label_durs'].append((label,15.0))
                    break
                if label != last_label or i==len(epgroup)-1:
                    last_dur = epgroup.iloc[[i-1]].label_dur.values[0]
                    if pruef:
                        if last_label != 1:
                            dur_dict[sp]['label_durs'].append((last_label,last_dur))
                    pruef = True
                last_label = label        
    for sp in range(2,8):
        lenfile = open(prepath+'/r'+str(sp)+'_sil_length.csv','w')
        lenfile.write(dur_dict[sp]['ep_num'])
        lenfile.write('\n')
        lenfile.writelines(dur_dict[sp]['label_durs'])
    lenfile.close()
    return dur_dict

#input: path to pickled episodes and a treshold
#return: the frequency dict of {sp: {Falsepositives(MTP labeled as TRP),(MTP+TRP)labeled_correctly, mtp_labeled_correctly}}
def baseline(episodes_path, models_path, threshold=.76):

    #create dictionary of silence durations and number of episodes
    #{ep_num:int, label_durs:[...]}
    modpath = models_path+'Baseline'
    if not os.path.isdir(modpath):
        os.makedirs(modpath)
            
    prepath = modpath+'/r'+str(speaker)
    if not os.path.isdir(prepath):
        os.makedirs(prepath)

    base_dict = baseline_dict(episodes_path, prepath, train=False)

    baseline_dict = {}

    for sp in range(2,8):
        lenfile = open(prepath+'/r'+str(sp)+'_sil_length.csv','r')
        lines = [float(line.strip()) for line in lenfile.readlines()]
        ep_num = lines[0].strip()
        label_durs = lines[1:]
        baseline_dict[sp] = {}
        lt750 = [dur for dur in label_durs]
        gt750 = [dur for dur in label_durs]
        base = (ep_num+len(lt750))/float(len(label_durs))
        fp = (len(gt750)-ep_num)/float(len(label_durs))
        mtp_base = len(lt750)/float(len(label_durs))

        baseline_dict[sp]['FP'] = fp
        baseline_dict[sp]['labeled_correctly'] = baseline
        baseline_dict[sp]['mtp_labeled_correctly'] = mtp_base
        

        print sp
        print 'False positives (MTP labeld as TRP)',fp*100
        print 'Threshold 750ms baseline\n (True positives, MTP and TRP that are labeled correctly) ',baseline*100
        print '\n'
        

    return baseline_dict

if __name__ == "__main__":
    episodes_path = "../../../eot_detection_data/pickled_episodes_old_andChanged_wav13sec_merged/"
    models_path = "../../../eot_detection_data/Models/"
    model_name = 'lstm_1'#'lstm' #or LogisticRegression, MMT
    ##########################################################
    #here you can set the values for the start and end speaker,
    #the modes you want to produce a pickled model for
    #and the treshold for the baseline
    #the default values are set to create all models and the baseline threshold at .76
    #that means the EOTaccuracy for a model that cuts-in after 750ms of silence
    ##########################################################
    #start = 2
    #end = 8
    #modes = ['acoustic','lm','acousticlm']
    #threshold = .76
    
    test_model_labels(model_name, episodes_path,models_path,\
                             start=2,end=8,\
                             modes=['acousticlm'])
    
    
    #print the baseline for a threshold
    #base_dict = baseline(episodes_path, models_path, threshold=.76)
