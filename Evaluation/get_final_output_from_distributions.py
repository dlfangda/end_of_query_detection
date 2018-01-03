from __future__ import division
#!python

#################################INFO####################################
# Script to take distribution outputs (numpy arrays) from a given model as input and output incremental output files for those episodes.
# These can be calculated by a Viterbi decode with a Markov Model or through less expensive means like thresholding.

import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict

from patsy import dmatrices
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support, classification_report
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from evaluation_utils import lsdir, write_inc_sequence_to_file

print "pandas version, needs to be 0.18+", pd.__version__
print "sklearn version, needs to be 0.18+", sklearn.__version__

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

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

    dev_test_episodes = open(THIS_DIR + '/dev_test_eps.csv','r')
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

def evaluate_episode(predictions, labels, c_window=75):
    # Acc: Accuracy of EOT detection in target c_window (final output)
    # TTD: For correctly detected EOTs in the window, what is the average time to detection
    # Cut-in rate: How often does the system cut in over all episodes
    # Latency: (can be negative) At what average point relative to the gold standard is EOT predicted
    # EO: Edit Overhead of labels to give stability- mean over all episodes
    correct = False
    TTD = None #can be None if cut-in has been committed to as the cut-ins don't count
    cut_in = False
    latency = None
    num_edits = 0
    final_output = []
    eot = 0
    predicted_eot = False

    for pred, label in zip(predictions, labels):
        if label == 2.0:
            eot+=1
        if len(pred) == 1:
            #single prediction, no backtracking
            final_output.extend(pred)
        else:
            final_output = final_output[:-(len(pred)-1)] + pred
        if final_output[-1] == 2.0: #latest prediction is a eot
            if label == 2.0 and not predicted_eot: #only allowing latency measures for correct predictions
                latency = eot-1 #i.e. how many frames have we had of eot in gold
                predicted_eot = True
                if eot>0 and eot <= c_window+1:
                    correct = True
          
                
        num_edits+=len(pred)
    i = 0
    #check whether cut-in remains in final output
    for pred,label in zip(final_output,labels):
        if int(pred) == 2 and not int(label) == 2:
            cut_in = True
            correct = False
            latency = None
            #might be able to get the latency here
            
            for j in range(i,len(labels)):
                if int(labels[j]) == 2:
                    negative_latency = i - j
                    print "neg latency cut in", negative_latency, "at", j, "from",len(labels)
                    if negative_latency > -5: print "not severe!"
                    break
            #for x,y in zip(final_output,labels):
            #    print x,y
            #raw_input()
            break
        i+=1
    EO = float(num_edits)/float(len(final_output))
    #if latency and not latency == 75:
    #    print latency
    #    for x,y in zip(final_output,labels):
    #        print x,y
    #    raw_input()
    return final_output, correct, TTD, cut_in, latency, EO
    
def classify_window(window,weight_on_trp):
    mtp_average = np.product([row[0] for row in window])
    s_average = np.product([row[1] for row in window])
    trp_average = np.product([weight_on_trp * row[2] for row in window])
    return np.argmax([mtp_average,s_average,trp_average])

def get_final_output_for_episode(ep_name,distributions,labels,
                                 mmt=None,
                                 back_off_limit =None,
                                 weight_on_trp = None,
                                 number_of_trps = None):
    #use a markov model, or simple thresholding technique to get outputs
    #numer_of_trps = the number of frames over which trp has to be overall max
    #if using gold VAD this only need be the last frame before window:
    window_size = 1 + number_of_trps 
    
    raw_predictions = []
    inc_predictions = []
    trp = 0
    pred_trp = 0
    correct = False
    cut_in = False
    window = [[1,0,0]] * window_size
    time_in_silence = 0 #this is for backing off to silence threshold
    speech_started = False
    for origdist,label in zip(distributions,labels):
        window = window[1:] #shift window across
        if label == 1.0: 
            dist = [0.0,1.0,0.0] #gold speeech- like VAD
        else:
            dist = deepcopy(origdist)
            dist[1] = 0.0 #no speech
            total = dist[0] + dist[2]
            #normalize the other two
            dist[0] = dist[0]/total
            dist[2] = dist[2]/total
        window.append(dist)
        if time_in_silence >= back_off_limit: #already seen the backoff limit
            pred = 2  
        elif classify_window(window[:-number_of_trps],weight_on_trp) == 1 and \
                classify_window(window[-number_of_trps:],weight_on_trp) == 2:
            pred = 2
        #elif classify_window(window[-number_of_trps:],weight_on_trp) == 2:
        #    pred = 2
        else:
            #pred = np.argmax(dist)
            if label == 1.0: #speech
                speech_started = True
                pred = 1.0
                time_in_silence = 0
            else: #if either silence category
                pred = 0
                if speech_started:
                    time_in_silence +=1
            
        #print pred
        inc_predictions.append([pred]) #nb last one just for testing

    return inc_predictions

def get_baseline(labels,threshold = 60):
    #print threshold
    #print type(threshold)
    inc_outputs = []
    threshold_reached = False
    speech_started = False
    silence = 0
    for l in labels:
        if l != 1.0 and speech_started:
            silence+=1
        elif l==1.0 and not speech_started:
            speech_started = True
        if silence > threshold:
            if not threshold_reached:
                inc_outputs.append((threshold + 1) * [2.0])
                threshold_reached = True
            else:
                inc_outputs.append([2.0]) #pad the end with eot
        elif l != 1.0: #treat as silence, mtp
            inc_outputs.append([0.0])
        else:
            inc_outputs.append([1.0]) #speech
            silence = 0 #resets threshold
    return inc_outputs
    

def get_final_output_for_fold(eppath,model_path, model_name,\
                              eps_test, heldout_speaker, \
                              mode, dataset_name,
                              back_off_limit = None,
                              weight_on_trp = None,
                              number_of_trps = None,
                              baseline=False):
    
    #create the inc_output path folder for this fold if needed
    inc_output_dir = model_path+'/r'+str(heldout_speaker)+'/inc_output'
    if not os.path.isdir(inc_output_dir):
        os.makedirs(inc_output_dir)
    
    print dataset_name,' set for speaker: ', heldout_speaker, "mode", mode

    ep_paths = lsdir(eppath+'/r'+str(heldout_speaker)+'/','pkl')
    
    filemode = mode if not baseline else "acoustic"
    fprob = model_path+'/r'+str(heldout_speaker)+'/distributions/prob_dist'+\
            '_r'+str(heldout_speaker)+'_'+filemode+'_'+dataset_name + ".pkl"

    with open(fprob, 'rb') as input_prob:
        prob_dists = pickle.load(input_prob)
    print len(prob_dists)
    limit = 10
    current_ep = 0
    current_speaker = 0
    current_dist = []
    current_labels = []
    scores =[]
    overall_predictions = []
    overall_labels = []
    overall_latencies = []
    num_correct = 0
    num_cut_in = 0
    num_eps = 0
  
    for r in range(0,len(prob_dists)):
        row = prob_dists[r]
        label, mtp, s, trp, ep_number, speaker = row
        #if limit > 0: 
        #    print label, mtp, s, trp, ep_number, speaker
        limit-=1
        if (not current_speaker == speaker) or (not current_ep == ep_number)\
            or (r == len(prob_dists)-1):
            #if speaker change or last row then get results for last speaker
            
            if (r == len(prob_dists)-1): #if last row add last row
                current_dist.append(deepcopy([mtp, s, trp]))
                current_labels.append(label)
            
            if not current_labels == []:
                #not the first time
                #print current_labels.count(2), current_labels.count(2.0)
                #print current_labels
                #raw_input()
                test = True
                trp_start = False
                for l in current_labels:
                    if l == 2.0 and not trp_start:
                        trp_start = True
                    elif l != 2.0 and trp_start:
                        test = False
                        break
                if current_labels.count(2) < 1000 or not test:
                    print "test illegal", test
                    print "or no final 2s"
                    current_ep = ep_number
                    current_speaker = speaker
                    current_dist = []
                    current_labels = []
                    continue
                num_eps+=1
                if baseline:
                    inc_predictions = get_baseline(current_labels,threshold = int(back_off_limit/10))
                        #print inc_predictions
                else:
                    inc_predictions = get_final_output_for_episode(
                                        str(ep_number),
                                        current_dist,
                                        current_labels,
                                        mmt=None, 
                                        back_off_limit = int(back_off_limit/10),
                                        weight_on_trp = weight_on_trp,
                                        number_of_trps = number_of_trps)
               

                predictions, correct, TTD, cut_in, latency, EO = evaluate_episode(inc_predictions,
                                                                                  current_labels)
                #for each file writing the predictions as an output array
                df = pd.DataFrame.from_dict(OrderedDict([ 
                      ('mtp',[ d[0] for d in current_dist]),
                      ('s', [ d[1] for d in current_dist]),
                      ('trp', [ d[2] for d in current_dist]),
                      ('prediction', predictions),
                      ('label',current_labels),
                      ('cut_in',[cut_in] * len(predictions)),
                      ('latency',[latency] * len(predictions))
                      ] ))
                inc_output_filepath = inc_output_dir + \
                                                      "/inc_output_r{}_{}_".format(current_speaker,
                                                                                   current_ep) + \
                                                        mode + ".pkl"
                with open(inc_output_filepath, 'wb') as output:  # Note: `ab` appends the data
                    pickle.dump(df, output, pickle.HIGHEST_PROTOCOL)
                p_r_f_tags = precision_recall_fscore_support(current_labels,
                                                             predictions, 
                                                             average='weighted')
                #for p, l in zip(predictions,current_labels):
                #    
                if correct: num_correct+=1
                if cut_in: num_cut_in+=1
                if latency: overall_latencies.append(latency)
                overall_predictions.extend(deepcopy(predictions))
                overall_labels.extend(deepcopy(current_labels))
                test_result = p_r_f_tags[2]
                scores.append(test_result)
            current_ep = ep_number
            current_speaker = speaker
            current_dist = []
            current_labels = []

        current_dist.append(deepcopy([mtp, s, trp])) #useless on last iteration
        current_labels.append(label)
                                                
    return overall_labels, overall_predictions, scores, num_eps, num_correct, num_cut_in, overall_latencies

def final_output_for_speaker_range(eppath=THIS_DIR + '/../../pickledEpisodes',\
              model_path=None,\
              model_name=None,\
              start=2, end=8 , subset='test',\
              modes=['acoustic','lm','acousticlm'],
                         back_off_limit=None,
                         weight_on_trp = None,
                         number_of_trps = None,
                         results_file_path="results.text"):

    dev_test_episodes = open(THIS_DIR + '/dev_test_eps.csv','r')
    eps = dev_test_episodes.readlines()
    dev_test_episodes.close()
    
    eps_dict = {}
    for line in eps:
        line = line.strip()
        line = line.split("\t")
        eps_dict[line[0]] = line[1].split(',')
    
    results_file = open(results_file_path,"a")
    print>>results_file, "%%%%%%%%%%%%%%%%%%%%%%%%%"
    for mode in modes:
        print mode
        overall_labels = []
        overall_predictions = []
        overall_scores = []
        overall_latencies = []
        overall_eps = 0
        overall_correct = 0
        overall_cut_in = 0
        for speaker in range(start, end):
            print "speaker", speaker
            fold_results_file = open(results_file_path.replace(".text","_r"+str(speaker)+".text"),"a")
            print>>fold_results_file, "%%%%%%%%%%%%%%%%%%%%%%%%%"
            if subset == 'test':
                eps_set =  eps_dict['test_'+str(float(speaker))]
                print 'number of episodes in test set:\n', len(eps_set)
            elif subset == 'dev':
                eps_set =  eps_dict['dev_'+str(float(speaker))]
                print 'number of episodes in dev set:\n', len(eps_set)
            labels, predictions, scores, num_episodes, num_correct, num_cut_in, latencies = get_final_output_for_fold(eppath, 
                                                                    model_path, 
                                                                    model_name, 
                                                                    eps_set, 
                                                                    speaker, 
                                                                    mode, 
                                                                    subset,                                                                                                                           back_off_limit = back_off_limit,
                                                                    weight_on_trp = weight_on_trp,
                                                                    number_of_trps = number_of_trps,
                                                                    baseline=mode=="baseline")
            overall_predictions.extend(deepcopy(predictions))
            overall_labels.extend(deepcopy(labels))
            overall_scores.extend(deepcopy(scores))
            overall_latencies.extend(deepcopy(latencies))
            overall_eps+=num_episodes
            overall_correct+=num_correct
            overall_cut_in+=num_cut_in
            print "Mode", mode, str(back_off_limit), str(weight_on_trp) + "_" + str(number_of_trps), str(weight_on_trp)
            print "speaker",speaker,"fold result",num_cut_in/num_episodes, np.average(latencies)
            print>>fold_results_file, "Mode", mode, str(back_off_limit) + "_" + str(number_of_trps), str(weight_on_trp)
            #print np.average(overall_scores), np.std(overall_scores)
            #print>>results_file, classification_report(overall_labels,overall_predictions,labels=
             #                                [0,1,2],target_names=['MTP','S','TRP'])
            print>>fold_results_file, float(num_correct)/float(num_episodes), num_correct, num_episodes,"correct in window"
            print>>fold_results_file, float(num_cut_in)/float(num_episodes), num_cut_in, num_episodes,"cut_ins"
            print>>fold_results_file, np.average(latencies), np.std(latencies), "latency"
            #fold_results_file.close()
            
        print>>results_file, "Mode", mode, str(back_off_limit) + "_" + str(number_of_trps), str(weight_on_trp)
        #print np.average(overall_scores), np.std(overall_scores)
        #print>>results_file, classification_report(overall_labels,overall_predictions,labels=
         #                                [0,1,2],target_names=['MTP','S','TRP'])
        print>>results_file, float(overall_correct)/float(overall_eps), overall_correct, overall_eps,"correct in window"
        print>>results_file, float(overall_cut_in)/float(overall_eps), overall_cut_in, overall_eps,"cut_ins"
        print>>results_file, np.average(overall_latencies), np.std(overall_latencies), "latency"
        #results_file.close()
   
    return

if __name__ == "__main__":
    #model_path = '../../Logistic_Regression_final'
    #model_path = '../../lstm_stateful_1_layer_68_nodes'
    #episodes_path = "../../../eot_detection_data/pickled_episodes_old_andChanged_wav13sec_merged/"
    #model_name = 'lstm_1'
    #model_path = "../../../eot_detection_data/Models/lstm_1"
    #model_path = '../../lstm_stateful_1_layer_68_nodes'
    
    
    model_name = 'lstm_5'
    #number_of_trps = 100
    model_path = THIS_DIR + "/../../../eot_detection_data/Models/" + model_name
    episodes_path = THIS_DIR + "/../../../eot_detection_data/Data/pickled_episodes/"
    #model_path = '../../lstm_stateful_1_layer_68_nodes
    results_file = THIS_DIR + "/Results/results_{}_folds.text".format(model_name)
    #baselines
    baseline = False
    if baseline:
        for bk in range(50,6001,50): #+ range(1000,6001,500):
            print "baseline", bk
            back_off_limit = bk
            weight_on_trp = 1.0 #none
            final_output_for_speaker_range(eppath=episodes_path,\
                                          model_path=model_path,\
                                          model_name=model_name,\
                                          start=2, end=8 , subset='dev',\
                                          modes=['baseline'],
                                          back_off_limit=back_off_limit,
                                          weight_on_trp = weight_on_trp,
                                          results_file_path= results_file)
        #quit()
    print "evaluating",model_name
    
    for w in [1.0]:
        for bk in range(50,2001,50) + range(2500,6001,500):
            #my_range = range(10,int((bk/10)*(2/3)),10) if bk>150 else [10]
            #ratio = 0.4 #in Interspeech paper we use this, works well
            #my_range = [int((bk/10)* ratio)]
            #if bk < 1000:
            #    my_range = range(2,int(bk/10),2) #try every other window
            #else:
            my_range = range(int((bk/10)*0.3),int((bk/10)*0.5),max([int((bk/10)*0.1),1]))
            if my_range == []:
                my_range = [int((bk/10)*0.5)]
            for number_of_trps in my_range:
                print "number of trps", number_of_trps
                print "bk",bk
                back_off_limit = bk
                weight_on_trp = w #none
                final_output_for_speaker_range(eppath=episodes_path,\
                                              model_path=model_path,\
                                              model_name=model_name,\
                                              start=2, end=8 , subset='dev',\
                                              modes=['acoustic','lm','acousticlm'],
                                              back_off_limit=back_off_limit,
                                              weight_on_trp = weight_on_trp,
                                              number_of_trps = number_of_trps,
                                              results_file_path= results_file
                                              )
