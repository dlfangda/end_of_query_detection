#!python

import os
import pickle
import codecs
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib

#args: rootdir -> root directory, ending -> file ending
#return: list of pathes in rootdir 
def lsdir(rootdir, ending):
    pathlist = []
    for root,  dirs,  files in os.walk(rootdir,  topdown=False):
        for filename in files:
            if ending in filename:
                pathlist.append(filename)
    return pathlist

def write_inc_sequence_to_file(ep, inc_file, inc_labels):

    inc_file.write(str(ep))
    inc_file.write('\t')
    ep_label_seq = [';'.join(label) if type(label) == list else label for label in inc_labels]
    inc_file.write(','.join(ep_label_seq))
    inc_file.write('\n')
      
    return

def get_baseline_sequences(eppath,model_path, model_name,\
                           threshold,\
                           eps_set, heldout_speaker, \
                           mode, dataset_name, fprep='inc_output'):
    acoustic_features = 'label ~ time_in_sec + \
pcm_RMSenergy_sma + rms_minus_one + rms_minus_two + rms_minus_three + rms_minus_four +\
pcm_LOGenergy_sma + pcm_loudness_sma + pcm_intensity_sma  +\
zscore + duration + intensity_mean + intensity_slope'
    ######################################################
    lm_features = 'label ~ time_in_sec +\
wml + wml_trigram'
    ######################################################
    acoustic_and_lmfeatures = acoustic_features + " + " +\
                              lm_features[lm_features.find('time_in_sec +')+13:]
        
    if not os.path.isdir(model_path+'/r'+str(heldout_speaker)+'/inc_output'):
        os.makedirs(model_path+'/r'+str(heldout_speaker)+'/inc_output')
        
    if mode == 'acoustic':
        features = acoustic_features
    elif mode == 'lm':
        features = lm_features
    elif mode == 'acousticlm':
        features = acoustic_and_lmfeatures

    print dataset_name,' set for speaker: ', heldout_speaker

    ep_paths = lsdir(eppath+'/r'+str(heldout_speaker)+'/','pkl')
        
    fprep += '_th'+str(int((threshold-0.01)*1000))
    inc_file = codecs.open(model_path+'/r'+str(heldout_speaker)+\
                           '/inc_output/'+fprep+'_'+str(heldout_speaker)+\
                           mode+'_'+dataset_name+'.csv','w','utf8')
    
    
    pointer=0
    for ep_path in ep_paths:
    
        ep = ep_path.split('_')[1].split(r'.')[0]
        sp = float(ep_path.split('_')[0][1])
        
        if not str(float(ep)) in eps_set: continue

        print sp,ep
        path = eppath+'/r'+str(heldout_speaker)+'/'+ep_path        
        
        with open(path,'rb') as pkl_file:
            test_ep = pickle.load(pkl_file)

        labels = test_ep['label'].values.tolist()
        durs = test_ep['label_dur'].values.tolist()
        print 'gold', len(labels)

        convert_dict = {0:'mtp',1:'speech',2:'trp'}
        cutin_seq = ['trp'+str(x*10) for x in range(1,6)]

        print 'get incremental baseline labels'

        inc_labels = []
        threshold_reached = False
        for i in range(len(labels)):
            
            label = convert_dict[int(labels[i])]
            
            #if durs[i] < .06:
            #    label += str(int(durs[i]*1000))
            #else:
            #    label += '50'

            #print label

            first_speech = False
                
            if not label.startswith('speech'):

                if first_speech == False:
                    inc_labels.append(label)
                else:
                    if durs[i] < threshold:
                        inc_labels.append('mtp')                    
                    else:
                        if threshold_reached == False:
                            threshold_reached = True
                            inc_labels.append(int(threshold*101)*['trp10'])
                        else:
                            inc_labels.append('trp10')

            else:
                if first_speech == False:
                    first_speech = True
                inc_labels.append(label)
                
        print 'inc',len(inc_labels)

        #print inc_labels
        #break

        write_inc_sequence_to_file(ep, inc_file, inc_labels)

        if pointer % 5 ==0 or int(ep)%5==0:
            print 'progress: ', pointer, ep

        pointer+=1
        
       
    inc_file.close()
        
        
    eps = None
    speaker_eps = None
    eps_dict = None

    return 

def get_baseline_for_speaker_range(eppath=r'./../../pickledEpisodes',\
              model_path=r'./../../Baseline',\
              model_name='Baseline',\
                                   threshold=.76,\
              start=2, end=8 , dataset_name='test',\
                                   modes=['acoustic','lm','acousticlm']):

    print 'create baseline sequences...'
    
    dev_test_episodes = open('./dev_test_eps.csv','r')
    eps = dev_test_episodes.readlines()
    dev_test_episodes.close()
    
    eps_dict = {}
    for line in eps:
        line = line.strip()
        line = line.split("\t")
        eps_dict[line[0]] = line[1].split(',')
        
    for speaker in range(start, end):
        print "speaker", speaker
        if dataset_name == 'test':
            eps_set =  eps_dict['test_'+str(float(speaker))]
            print 'number of episodes in test set:\n', len(eps_set)
        elif dataset_name == 'dev':
            eps_set =  eps_dict['dev_'+str(float(speaker))]
            print 'number of episodes in dev set:\n', len(eps_set)
        
        for mode in modes:
            print mode
            get_baseline_sequences(eppath,model_path, model_name,\
                           threshold,\
                           eps_set, speaker, \
                           mode, dataset_name, fprep='inc_output')

    return

if __name__ == "__main__":
    model_name = 'Baseline'
    model_path = r'./../../Baseline'
    get_baseline_for_speaker_range(eppath=r'./../../pickledEpisodes',\
                                  model_path=model_path,\
                                  model_name=model_name,\
                                threshold=.61,\
                                  start=2, end=3 , dataset_name='test',\
                                   modes=['lm'])
                                  #modes=['acoustic','lm','acousticlm'])    
