#!python

import os
import pickle
import numpy as np
import pandas as pd
from patsy import dmatrices
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from evaluation_utils import round_down, lsdir
from EOT_features import acoustic_features, lm_features

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

print "pandas version, needs to be 0.18+", pd.__version__
print "sklearn version, needs to be 0.18+", sklearn.__version__

def get_dev(heldout, prepath):
    print 'dev set for speaker: ', heldout
    
    ep_pathes = lsdir(prepath+'r'+str(heldout)+'/','pkl') 
    
    dev = pd.DataFrame()

    dev_test_episodes = open(THIS_DIR + '/dev_test_eps.csv','r')
    eps = dev_test_episodes.readlines()
    dev_test_episodes.close()
    
    eps_dict = {}
    for line in eps:
        line = line.strip()
        line = line.split("\t")
        eps_dict[line[0]] = line[1].split(',')
    
    eps_dev =  eps_dict['dev_'+str(float(heldout))]
    print 'number of episodes in dev set:\n', len(eps_dev)

    for ep_path in ep_pathes:
        ep = ep_path.split('_')[1].split(r'.')[0]
        if (float(ep)/len(ep_pathes))/10 == 0:
            print ep
        #print ep_path
        path = prepath+'r'+str(heldout)+'/'+ep_path
        pkl_file = open(path,'rb')
        if str(float(ep)) in eps_dev:
            dev_ep = pickle.load(pkl_file)
            dev = pd.concat([dev, dev_ep])
        else:
            continue
        pkl_file.close()
        
    eps = None
    speaker_eps = None
    eps_dict = None
    
    return dev

def get_test(heldout, prepath):

    print 'test set for speaker: ', heldout
    
    ep_pathes = lsdir(prepath+'r'+str(heldout)+'/','pkl') 
    
    test = pd.DataFrame()

    dev_test_episodes = open(THIS_DIR + '/dev_test_eps.csv', 'r')
    eps = dev_test_episodes.readlines()
    dev_test_episodes.close()
    
    eps_dict = {}
    for line in eps:
        line = line.strip()
        line = line.split("\t")
        eps_dict[line[0]] = line[1].split(',')
    
    eps_test =  eps_dict['test_'+str(float(heldout))]
    print 'number of episodes in test set:\n', len(eps_test)

    for ep_path in ep_pathes:
        ep = ep_path.split('_')[1].split(r'.')[0]
        if (float(ep)/len(ep_pathes))/10 == 0:
            print ep
        print ep_path
        path = prepath+'r'+str(heldout)+'/'+ep_path
        pkl_file = open(path,'rb')
        if str(float(ep)) in eps_test:
            test_ep = pickle.load(pkl_file)
            test = pd.concat([test, test_ep])
        else:
            continue
        pkl_file.close()
        
    eps = None
    speaker_eps = None
    eps_dict = None

    return test

def get_prob_dist_for_episode(sp, ep, labels, model, X, model_name):    

    if "Logistic_Regression" in model_name:
        prob_dists = model.predict_proba(X)
    else:
        seq_len = 10
        feature_len = X.shape[2]
        print X.shape
        X = np.reshape(X, (X.shape[0], seq_len, feature_len))
        totalTimeSteps = len(X)
        X = X[:round_down(totalTimeSteps, 10)] # for batches need to round down
        prob_dists = model.predict_proba(X,batch_size=10, verbose=1)
        print "---"
        print prob_dists
        print prob_dists.shape
        #pad the end with the same final prediction to make it the right dimension
        difference_in_steps = totalTimeSteps - len(prob_dists)
        difference = [prob_dists[-1]] * difference_in_steps
        print difference, "diff"
        if len(difference) > 0: # make up the difference
            prob_dists =  np.append(prob_dists, difference, axis=0)
        print prob_dists.shape, "new shape with diff"
        model.reset_states() #in stateful reset the 
    
    prob_dists = np.insert(prob_dists,len(prob_dists[0]),[ep], axis=1)
    prob_dists = np.insert(prob_dists,len(prob_dists[0]),[sp], axis=1)
    prob_dists = np.insert(prob_dists,0,[labels], axis=1)
      
    return prob_dists

def get_prob_distributions(eppath,model_path, model_name,\
                           eps_test, heldout_speaker, \
                           mode, dataset_name, fprep='prob_dist'):

    acoustic_and_lmfeatures = acoustic_features + " + " +\
                        lm_features[lm_features.find("wml "):]
        
    if not os.path.isdir(model_path+'/r'+str(heldout_speaker)+'/distributions'):
        os.makedirs(model_path+'/r'+str(heldout_speaker)+'/distributions')

    if mode == 'acoustic':
        features = acoustic_features
    elif mode == 'lm':
        features = lm_features
    elif mode == 'acousticlm':
        features = acoustic_and_lmfeatures

    print dataset_name,' set for speaker: ', heldout_speaker

    ep_paths = lsdir(eppath+'/r'+str(heldout_speaker)+'/','pkl')
    
    fprob = model_path+'/r'+str(heldout_speaker)+'/distributions/'+fprep+\
            '_r'+str(heldout_speaker)+'_'+mode+'_'+dataset_name + ".pkl"
    
    ext = ".pkl" if 'Logistic_Regression' in model_name else '.h5'
    fname = model_path+'/r'+str(heldout_speaker)+'/'+str(heldout_speaker)+\
            r'_' + mode + ext
    print fname
    if "Logistic_Regression" in model_name:
        model = joblib.load(fname)
    else:
        model = load_model(fname)
    
    init_array = False
    i=0
    
    scaler_path = model_path+'/r'+str(heldout_speaker)+\
                      '/'+str(heldout_speaker)+'_'+mode+'_scaler.pkl'
    with open(scaler_path, 'rb') as scaler_pkl:
        scaler = pickle.load(scaler_pkl)
    
    prob_dists = None
    for ep_path in ep_paths:
    
        ep = ep_path.split('_')[1].split(r'.')[0]
        sp = float(ep_path.split('_')[0][1])
        if not str(float(ep)) in eps_test: continue
        
        path = eppath+'/r'+str(heldout_speaker)+'/'+ep_path        
        
        pkl_file = open(path,'rb')
        test_ep = pickle.load(pkl_file)

        labels = test_ep['label'].values.tolist()
        
        y, X = dmatrices(features, test_ep, return_type="dataframe")
        X = scaler.transform(X)
        y = np.ravel(y)
        if len(X) == 0:
            print "leaving out null episode"
            continue
        if not "Logistic_Regression" in model_name:
            #for lstms/sequential models we need to pad it.
            seq_len = 10
            fold_dataX = []
            for i in range(0,X.shape[0]):
                fold_dataX.append(X[i-seq_len:i])
            X = pad_sequences(fold_dataX, maxlen=seq_len, dtype='float32')
            #print "fold with padding new shape", X.shape

        if not init_array:
            prob_dists = get_prob_dist_for_episode(sp, ep, y, model, X, model_name)
            init_array = True
        else:
            ep_prob_dists = get_prob_dist_for_episode(sp, ep, y, model, X, model_name)
            prob_dists = np.append(prob_dists, ep_prob_dists, axis=0)
            
        if i % 5 ==0 or int(ep)%5==0:
            print 'progress: ', i, ep
        i+=1
        
       
        pkl_file.close()
        
    print len(prob_dists),"rows"
    with open(fprob, 'wb') as output:  # Note: `ab` appends the data
        pickle.dump(prob_dists, output, pickle.HIGHEST_PROTOCOL)
        
    #eps = None
    #speaker_eps = None
    #eps_dict = None
    return 

def prob_dist_for_speaker_range(eppath=r'./../../pickledEpisodes',\
              model_path=r'./../../Logistic_Regression',\
              model_name='Logistic_Regression',\
              start=2, end=8 , subset='test',\
              modes=['acoustic','lm','acousticlm']):

    dev_test_episodes = open(THIS_DIR + '/dev_test_eps.csv','r')
    eps = dev_test_episodes.readlines()
    dev_test_episodes.close()
    
    eps_dict = {}
    for line in eps:
        line = line.strip()
        line = line.split("\t")
        eps_dict[line[0]] = line[1].split(',')
        
    for speaker in range(start, end):
        print "speaker", speaker
        if subset == 'test':
            eps_set =  eps_dict['test_'+str(float(speaker))]
            print 'number of episodes in test set:\n', len(eps_set)
        elif subset == 'dev':
            eps_set =  eps_dict['dev_'+str(float(speaker))]
            print 'number of episodes in dev set:\n', len(eps_set)
        
        for mode in modes:
            print mode
            get_prob_distributions(eppath, model_path, model_name, eps_set, speaker, mode, subset, fprep='prob_dist')

    return

if __name__ == "__main__":
    episodes_path = THIS_DIR + "/../../../eot_detection_data/Data/pickled_episodes/"
    model_name = 'lstm_5'
    model_path = THIS_DIR + "/../../../eot_detection_data/Models/" + model_name
    #model_path = '../../lstm_stateful_1_layer_68_nodes'
    
    prob_dist_for_speaker_range(eppath=episodes_path,\
                                  model_path=model_path,\
                                  model_name=model_name,\
                                  start=7, end=8, subset='dev',\
                                  modes=['acoustic','lm','acousticlm'])
