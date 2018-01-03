#!python

import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report,\
     precision_recall_fscore_support
import pandas as pd
import codecs
import sys
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt


def round_down(num, divisor):
    return num - (num%divisor)

def lsdir(rootdir, ending):
    pathlist = []
    for root,  dirs,  files in os.walk(rootdir,  topdown=False):
        for filename in files:
            if ending in filename:
                pathlist.append(filename)
    return pathlist

def write_inc_sequence_to_file(inc_output_file,inc_predictions,distributions,labels):
    #saves a pickled dataframe to inc_output_file with the below rows
    #0_prob,1_prob,2_prob,argmax,output,label
    
    
    #old version
    #inc_file.write(str(ep))
    #inc_file.write('\t')
    #ep_label_seq = [';'.join(label) if type(label) == list else label for label in inc_labels]
    #inc_file.write(','.join(ep_label_seq))
    #inc_file.write('\n')
      
    return

####################################################################
#########                                               ############
#########               LOAD  system output             ############
#########               into dictionary                 ############
####################################################################
def load_pdist_df(path, ep_path):
    print 'load predictions from Logistic Regression and gold labels'
    with open(path, 'rb') as pkl_dist:
        pdist = pickle.load(pkl_dist)

    pred_labels = []

    for i in range(len(pdist)):#len(unpkl_dist):
        if i < len(pdist):
            slice_np = pdist[i:i+1,1:4]
            val = np.argmax(slice_np, axis=1)
            if len(val) > 1:
                print 'more than one label'
            else:
                val = val[0]
                #print slice_np, val
                pred_labels.append(val)

    pdist = np.insert(pdist,len(pdist[0]), np.array(pred_labels),axis=1)

    pdist_df = pd.DataFrame({'gold_label':pdist[:,0],'mtp':pdist[:,1],\
                       'speech':pdist[:,2], 'trp':pdist[:,3],\
                       'episode':pdist[:,4], 'speaker':pdist[:,5],\
                       'pred_label':pdist[:,6]})

    return pdist_df 

def load_dict(prepath='./../../Logistic_Regression', start=2,end=8, mode='lm',subset='test'):
    ep_dict = {}
    for speaker in range(int(start),int(end)):
        fname = prepath+'/r'+str(speaker)+'/distributions/prob_dist_r'+str(speaker)+'_'+mode+'_'+subset+'.pkl'
        ep_path = './../../pickledEpisodes/r'+str(speaker)+'/r'+str(speaker)+'_'
        pdist_df = load_pdist_df(fname, ep_path)#pd.read_csv(fname) 
        convert_dict = {2.0:'trp50',1.0:'speech50',0.0:'mtp50'}
        ep_dict[float(speaker)] = {}
        for ep in pdist_df.episode.unique():
            label_seq = list(pdist_df[pdist_df['episode']==ep].pred_label.values)
            label_seq = [convert_dict[label] for label in label_seq]
            ep_dict[float(speaker)][float(ep)] = label_seq
    return ep_dict

#return dict with incremental results per speaker for each episode
def load_inc_dict(prepath='./../../MMT',start=2,end=8,mode='lm',subset='test'):
    ep_dict = {}
    for speaker in range(start,end):
        dict_file = codecs.open(prepath+'/r'+str(speaker)+'/viterbiincremental'+str(speaker)+mode+'_'+subset+'.csv','r','utf8')
        ep_dict[float(speaker)] = {}
        while True:
            line = dict_file.readline()
            if line == '':
                break
            line_list = line.split('\t')
            ep_dict[float(speaker)][float(line_list[0])] = [label.split(';') if ';' in label else label for label in line_list[1].strip().split(',')]
        dict_file.close()
    return ep_dict

#return dict with final output label sequences for one speaker for the episodes
def load_final_out_dict(prepath='./../../MMT',start=2,end=8,mode='lm',subset='test'):
    inc_dict = load_inc_dict(prepath,start,end,mode,subset)
    ep_dict = {}
    for speaker in inc_dict:
        ep_dict[speaker] = {}
        for key,val in inc_dict[float(speaker)].iteritems():
            nvals = []
            for elem in val:
                if type(elem) == list:
                    nvals = nvals[:-(len(elem)-1)]
                    for ele in elem:
                        nvals.append(ele)
                else:
                    nvals.append(elem)
            ep_dict[speaker][key] = nvals 
    return ep_dict

####################################################################
#########                                               ############
#########               convert Labels                  ############
#########               for plotting                    ############
####################################################################

#to get smoothed values for one episode
#new values for plotting for the labels with time
#args: the episode id, pdist_df, 
def get_pdist_smooth(t_ep, pdist_df, ep_seq_dict):
    #ind_conv = {'mtp10': 0,
    ind_conv = {'mtp10':0.0,\
                       'mtp20': 0.0,\
                       'mtp30': 0.0,\
                       'mtp40': 0.0,\
                       'mtp50': 0.0,\
                       'se': 2.0,\
                       'speech10': 1.0,\
                       'speech20': 1.0,\
                       'speech30': 1.0,\
                       'speech40': 1.0,\
                       'speech50': 1.0,\
                       'trp10': 2.0,\
                       'trp20': 2.0,\
                       'trp30': 2.0,\
                       'trp40': 2.0,\
                       'trp50': 2.0}
    ep_stat_convert = [float(ind_conv[lab]) for lab in list(ep_seq_dict[t_ep])]
    lab_num = {lab:float(ind_conv[lab]) for lab in list(set(ep_seq_dict[t_ep]))} 
    for x in sorted([(num,str(val)[:4]) for num,val in lab_num.iteritems()]):
        print x
    ep_to_show = t_ep
    pdist_df_smoothed = pdist_df[pdist_df['episode']==ep_to_show]
    pdist_df_smoothed['smooth'] = ep_stat_convert
    return pdist_df_smoothed

#to get smoothed values for one episode
#new values for plotting for the labels with time
#args: the episode id, pdist_df, ep_dict
def get_pdist_smooth_converted(t_ep, pdist_df, ep_seq_dict):
    #ind_conv = {'mtp10': 0,
    ind_conv = {'mtp10':-1.2,\
                       'mtp20': -1.4,\
                       'mtp30': -1.6,\
                       'mtp40': -1.8,\
                       'mtp50': 0.0,\
                       'se': 2.5,\
                       'speech10': 0.2,\
                       'speech20': 0.4,\
                       'speech30': 0.6,\
                       'speech40': 0.8,\
                       'speech50': 1.0,\
                       'trp10': 1.2,\
                       'trp20': 1.4,\
                       'trp30': 1.6,\
                       'trp40': 1.8,\
                       'trp50': 2.0}
    ep_stat_convert = [float(ind_conv[lab]) for lab in list(ep_seq_dict[t_ep])]
    lab_num = {lab:float(ind_conv[lab]) for lab in list(set(ep_seq_dict[t_ep]))} 
    for x in sorted([(num,str(val)[:4]) for num,val in lab_num.iteritems()]):
        print x
    ep_to_show = t_ep
    pdist_df_smoothed = pdist_df[pdist_df['episode']==ep_to_show]
    pdist_df_smoothed['smooth'] = ep_stat_convert
    return pdist_df_smoothed

#to get smoothed values for one episode
#new values for plotting for the labels with time
#args: the episode id, pdist_df, 
def get_inc_pdist_smooth(t_ep, pdist_df, ep_seq_dict):
    #ind_conv = {'mtp10': 0,
    ind_conv = {'mtp10':-1.2,\
                       'mtp20': -1.4,\
                       'mtp30': -1.6,\
                       'mtp40': -1.8,\
                       'mtp50': 0.0,\
                       'se': 2.5,\
                       'speech10': 0.2,\
                       'speech20': 0.4,\
                       'speech30': 0.6,\
                       'speech40': 0.8,\
                       'speech50': 1.0,\
                       'trp10': 1.2,\
                       'trp20': 1.4,\
                       'trp30': 1.6,\
                       'trp40': 1.8,\
                       'trp50': 2.0}
    #################################
    ep_stat_convert = ep_seq_dict[t_ep]
    pre_ep_stat = []
    inc_time = []
    for i in range(len(ep_stat_convert)):
        if type(ep_stat_convert[i])==list:
            for l in ep_stat_convert[i]:
                pre_ep_stat.append(float(ind_conv[l]))
                inc_time.append((i+1)*0.01)
        else:
            pre_ep_stat.append(float(ind_conv[ep_stat_convert[i]]))
            inc_time.append((i+1)*0.01)
    ep_stat_convert = pre_ep_stat
    #print len(ep_stat_convert)
    #print len(inc_time)
    return inc_time, ep_stat_convert

####################################################################
#########                                               ############
#########               plotting                        ############
#########               functions                       ############
####################################################################

def plot_final_labels(speak,ep_to_show,pdist_df, plegend=False, smoothed=False):
    y = pd.Series(pdist_df[pdist_df['speaker']==speak][pdist_df['episode']==ep_to_show].pred_label)
    z = pd.Series(pdist_df[pdist_df['speaker']==speak][pdist_df['episode']==ep_to_show].gold_label)
    x = pd.Series(pdist_df[pdist_df['speaker']==speak][pdist_df['episode']==ep_to_show].time)
    smoo = pd.Series(pdist_df[pdist_df['speaker']==speak][pdist_df['episode']==ep_to_show].smooth)
    
    #print len(wm)
    #print len(z)
    ax = plt.subplot(111)
    #ax.set_xlim([0,25])
    ax.set_ylim([-.1,2.5])
    #ax.set_xlim([-0.5,35])
    plt.plot(x,y,lw=1)
    plt.plot(x,z,lw=.7)
    if smoothed:
        plt.plot(x,smoo,lw=.5)
    plt.xlabel('time in sec')
    plt.ylabel('label')
    plt.title('label sequence for speaker '+str(speak)+' episode '+str(ep_to_show))
    
    plt.yticks( [0.0, 1.0, 2.0],
        [r'MTP', r'Speech', r'TRP'])
    if plegend:
        if smoothed:
            plt.legend(['LR','gold','MMT'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            plt.legend(['LR','gold'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.show()

#plot one episode
#HMM smoothed label sequence (incremental)
#gold label
#predicted label sequence from logistic regression
def plot_an_episodes_final_label_sequence(sp, episode, prepath='./../../MMT', mode='lm', subset='test', legend=True, smoothed=True):
    print 'plot gold, pred and smoothed label sequence for speaker',sp,'episode',episode
    #########load dicts##########
    ep_dict = load_inc_dict(prepath,int(sp),int(sp)+1, mode, subset)
    ep_seq_dict = deepcopy(ep_dict[float(sp)])
    #print ep_seq_dict[27]
    print sorted(ep_seq_dict.keys())
    fname = './../../Logistic_Regression/r'+str(sp)+'/distributions/prob_dist_r'+str(sp)+'_'+mode+'_'+subset+'.pkl'
    ep_path = './../../pickledEpisodes/r'+str(sp)+'/r'+str(sp)+'_'
    pdist_df = load_pdist_df(fname, ep_path)
    #pdist_df = pd.read_csv(fname)
    #print pdist_df[pdist_df['speaker']==sp][pdist_df['episode']==episode]
    
    #############################
    #pdist_df_smoothed = get_pdist_smooth(episode, pdist_df, ep_seq_dict)
    inc_time, inc_seq = get_pdist_smooth(episode, pdist_df, ep_seq_dict)
    
    #pdist_df_smoothed.episode.unique()
    
    plot_labels(sp,episode, inc_time, inc_seq, pdist_df, legend, smoothed)
    
    
    for i in range(len(ep_seq_dict[episode])):
        if 'trp10' in ep_seq_dict[episode][i]:
            for inc in ep_seq_dict[episode][i-5:i]:
                print inc
            print (i+1)*0.01
            for inc in ep_seq_dict[episode][i:i+15]:
                print inc
            print '\n'
    
    return

def plot_inc_labels(speak,ep_to_show, inc_time, inc_seq, pdist_df):
    z = pd.Series(inc_seq)
    x = pd.Series(inc_time)
    
    pred = pd.Series(pdist_df[pdist_df['speaker']==speak][pdist_df['episode']==ep_to_show].pred_label)
    time_pred = pd.Series(pdist_df[pdist_df['speaker']==speak][pdist_df['episode']==ep_to_show].time)
    
    #print len(wm)
    #print len(z)
    ax = plt.subplot(111)
    ax.set_xlim([-1.0,20])
    ax.set_ylim([-1.0,2.5])
    #ax.set_xlim([-0.5,35])
    #plt.plot(x,y)
    plt.plot(x,z, 'b-')
    plt.plot(time_pred,pred,'r--')
    #plt.plot(x,smoo)
    
    plt.title("Incremental label sequence for speaker "+str(speak)+" episode "+str(ep_to_show))
    plt.xlabel("time in seconds")
    plt.ylabel("label MTP:0, S:1, TRP:2")
    plt.legend(['incremental label sequence','logistic regression'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

#plot one episode
#HMM smoothed label sequence
#gold label
#predicted label sequence from logistic regression
def plot_an_episodes_inc_label_sequence(sp, episode, prepath='./../../MMT', mode='lm', subset='test'):
    print 'plot gold, pred and smoothed label sequence for speaker',sp,'episode',episode
    #########load dicts##########
    ep_dict = load_inc_dict(prepath,int(sp),int(sp)+1, mode, subset)
    ep_seq_dict = deepcopy(ep_dict[float(sp)])
    #print ep_seq_dict[27]
    print sorted(ep_seq_dict.keys())
    fname = './../../Logistic_Regression/r'+str(sp)+'/distributions/prob_dist_r'+str(sp)+'_'+mode+'_'+subset+'.pkl'
    ep_path = './../../pickledEpisodes/r'+str(sp)+'/r'+str(sp)+'_'
    pdist_df = load_pdist_df(fname, ep_path)
    
    
    inc_time, inc_seq = get_inc_pdist_smooth(episode, pdist_df, ep_seq_dict)
    
    #pdist_df_smoothed.episode.unique()
    
    plot_inc_labels(sp, episode, inc_time, inc_seq, pdist_df)
    
    
    for i in range(len(ep_seq_dict[episode])):
        if 'trp10' in ep_seq_dict[episode][i]:
            for inc in ep_seq_dict[episode][i-5:i]:
                print inc
            print (i+1)*0.01
            for inc in ep_seq_dict[episode][i:i+15]:
                print inc
            print '\n'
    
    return

####################################################################
#########                                               ############
#########               Histograms                      ############
#########                                               ############
####################################################################

#returns a dictionary with the episodes as keys, for each episode
#the gold standard detection time for the TRP,
#the time the TRP is predicted with the logistic regression model
#the time the TRP is precicted with the MMT
def get_fd_dict(ep_seq_dict,pdist_df):
    first_detect = {}
    for ep in ep_seq_dict:
        try:
            time_smooth = ep_seq_dict[ep].index('trp10')*0.01
        except:
            time_smooth = -1
        try:
            time_pred = pdist_df[pdist_df['episode']==ep][pdist_df['pred_label']==2.0].time.values[0]
        except:
            time_pred = -1
        try:
            time_gold = pdist_df[pdist_df['episode']==ep][pdist_df['gold_label']==2.0].time.values[0]    
        except:
            time_gold = -1
        first_detect[ep] = {'ts':'{:.2f}'.format(time_smooth),\
                            'tp':'{:.2f}'.format(time_pred),\
                            'tg':'{:.2f}'.format(time_gold)}
    return first_detect

#returns a dictionary {sp:[temporal difference to gold TRP(tg)]}
#tp : time from LR
#ts : time from MMT
def get_temporal_diff_distribution(start=2, end=8, comp_time = 'tp', mode='lm',subset='test'):
    
    time_dist_dict = {}
    out = []
    
    for sp in range(start, end): 
        print sp
        time_dist_dict[sp] = []

        ep_dict = load_inc_dict(prepath,int(sp),int(sp)+1, mode, subset)
        ep_seq_dict = deepcopy(ep_dict[float(sp)])
        
        fname = './../../Logistic_Regression/r'+str(sp)+'/distributions/prob_dist_r'+str(sp)+'_'+mode+'_'+subset+'.pkl'
        ep_path = './../../pickledEpisodes/r'+str(sp)+'/r'+str(sp)+'_'
        pdist_df = load_pdist_df(fname, ep_path)
        
        first_detect = get_fd_dict(ep_seq_dict, pdist_df)
        
        for k,v in first_detect.iteritems():
            if float(v['tg']) == -1:
                out.append((k,v))
                continue
            elif float(v[comp_time]) == -1:
                diff = 15.0
                time_dist_dict[sp].append(diff)
            else:
                if float(v[comp_time]) < float(v['tg']):
                    diff = '{:.2f}'.format(float(v['tg'])-float(v[comp_time]))                     
                    diff = -abs(float(diff))
                elif float(v[comp_time]) == float(v['tg']):
                    diff = 0.0
                else:
                    diff = '{:.2f}'.format(float(v[comp_time])-float(v['tg']))
                    diff = abs(float(diff))
                time_dist_dict[sp].append(diff)  
    
    return time_dist_dict

def plot_temporal_difference_distribution(speaker, t_dist_dict, t_dist_dict_ts, title, bins_tp=60, bins_ts=17, for_all=False, plegend=True):
    if for_all:
        time_list = ['{:.2f}'.format(t) for speak in t_dist_dict for t in t_dist_dict[speak]]
        time_list_ts = ['{:.2f}'.format(t) for speak in t_dist_dict_ts for t in t_dist_dict_ts[speak]]
        print 'times all speakers', len(time_list)
        #print time_list
    else:
        time_list = ['{:.2f}'.format(t) for t in t_dist_dict[speaker]]
        time_list_ts = ['{:.2f}'.format(t) for t in t_dist_dict_ts[speaker]]
        print 'times speaker', speaker, len(time_list)
        #print time_list
        #print time_list_ts
    
    time_list = sorted([float(t) for t in time_list])
    time_list_ts = sorted([float(t) for t in time_list_ts]) 
    
    mean_tp = np.mean(time_list)
    mean_ts = np.mean(time_list_ts)
    print mean_tp, mean_ts
    
    std_tp = np.array(time_list).std()
    std_ts = np.array(time_list_ts).std()
    print std_tp, std_ts
    
    ymax = 100
    
    #if len(time_list) != 0:
    plt.hist(time_list_ts, bins=bins_ts, histtype='stepfilled', weights=[ymax/float(len(time_list))]*len(time_list), color='b', label='MMT')
    #std ts
    plt.plot([mean_ts-std_ts, mean_ts-std_ts], [0.0, ymax], color='b', linestyle='--', linewidth=.5, label='sd MMT')
    plt.plot([mean_ts+std_ts, mean_ts+std_ts], [0.0, ymax], color='b', linestyle='--', linewidth=.5)
    
    plt.hist(time_list, bins=bins_tp, histtype='stepfilled', weights=[ymax/float(len(time_list))]*len(time_list), alpha=0.5, color='r', label='LG')
    #std tp
    plt.plot([mean_tp-std_tp, mean_tp-std_tp], [0.0, ymax], color='r', linestyle='--', linewidth=.5, label='sd LG')
    plt.plot([mean_tp+std_tp, mean_tp+std_tp], [0.0, ymax], color='r', linestyle='--', linewidth=0.5)
    
    plt.ylim(0,ymax)
    plt.xlim(-25.0,20.0)
    if title != '':
        plt.title(title)
    elif for_all:
        plt.title("Histogram of temporal difference all speakers")
    else:
        plt.title("Histogram of temporal difference speaker "+str(speaker))
    plt.xlabel("temporal difference to gold standard in seconds")
    plt.ylabel("Frequency")
    if plegend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    return

####################################################################
#########               MMT                             ############
#########               10ms-window                     ############
#########               label                           ############
#########               Accuracy;                       ############
#########               Precision, Recall, F1score      ############
####################################################################

#compare two arrays gold and predicted
#for logistic regression per feature set model='LR'
#for MMT per feature set model='MMT'
#for dev or test set
#returns two dictionaries
#for accuracy: {speaker:{mode:accuracy_score_for_labels}}
#and classification report
def get_accuracy_precision_recall(model='MMT',\
                                  prepath='./../../MMT',\
                                  start=2, end=8,\
                                  modes=['acoustic','lm','acousticlm'],\
                                  subset='test'):

    accuracy_dict = {}
    classification_dict = {}
    pred_all = {}
    expect_all = {}
    
    for speaker in range(start, end):
        print speaker
        classification_dict[speaker] = {}
        accuracy_dict[speaker] = {}
        for mode in modes:
            print 'accuracy and classification report for speaker',str(speaker),'with',mode,'features:'
            model_seq = []
            gold_seq = []
            if not mode in pred_all:
                pred_all[mode] = []
            if not mode in expect_all:
                expect_all[mode] = []

            fname = './../../Logistic_Regression/r'+str(speaker)+'/distributions/prob_dist_r'+str(speaker)+'_'+mode+'_'+subset+'.pkl'
            ep_path = './../../pickledEpisodes/r'+str(speaker)+'/r'+str(speaker)+'_'
            pdist_df = load_pdist_df(fname, ep_path)
            if model == 'MMT':
                ep_seq_dict = load_final_out_dict(prepath, int(speaker), int(speaker)+1, mode, subset)
                print sorted(ep_seq_dict.keys())
                ep_dict = deepcopy(ep_seq_dict[float(speaker)])    
                
                for ep in ep_dict:
                    predLabels = []
                    for label in ep_dict[ep]:
                        if label.startswith('mtp'):
                            predLabels.append(0)
                        elif label.startswith('speech'):
                            predLabels.append(1)
                        else:
                            predLabels.append(2)
                        
                    model_seq += predLabels
                    gold_seq += list(pdist_df[pdist_df['speaker']==speaker][pdist_df['episode']==ep].gold_label.values)
                
            elif model == 'Logistic_Regression':
                for ep in list(pdist_df.episode.unique()):
                    model_seq += list(pdist_df[pdist_df['speaker']==speaker][pdist_df['episode']==ep].pred_label.values)
                    gold_seq += list(pdist_df[pdist_df['speaker']==speaker][pdist_df['episode']==ep].gold_label.values)

            model_seq = np.array(model_seq)
            gold_seq = np.array(gold_seq)

            #predictions = pdist_df[pdist_df['speaker']==speaker].pred_label 
            #expected = pdist_df[pdist_df['speaker']==speaker].gold_label
            #model = joblib.load('./pickledModels/r'+str(speaker)+'/log_res'+str(speaker)+mode+'.pkl')
            
            #classification_dict[speaker][mode] = classification_report(gold_seq, model_seq)
            p_r_f_tags = precision_recall_fscore_support(gold_seq, model_seq, average='weighted')
            test_result = p_r_f_tags[2]
            tag_summary =  classification_report(gold_seq, model_seq,\
                                                 labels=[0,1,2],\
                                                 target_names=['MTP','S','TRP'])
            accuracy_dict[speaker][mode] = accuracy_score(gold_seq, model_seq)
            classification_dict[speaker][mode] = (test_result,tag_summary)
            
            pred_all[mode] += list(gold_seq)
            expect_all[mode] += list(model_seq)
    
    #classification_dict['all'] = {}
    
    #for mode in modes:
    #    classification_dict['all'][mode] = classification_report(expect_all[mode], pred_all[mode])
                
    return accuracy_dict, classification_dict

def print_accuracy_class_report(accuracy_dict, classification_dict,\
                                start=2, end=8, mode='lm', subset='test'):
    print 'Accuracy and classification report per speaker for feature set: ',mode
    for speaker in range(start, end):
        print '\n',speaker
        print 'accuracy: ', accuracy_dict[speaker][mode]
        print 'f1-score: ',
        print classification_dict[speaker][mode][0]
        print 'classification report:'
        print classification_dict[speaker][mode][1]

    #print 'classification report over all speakers:'
    #print classification_dict['all'][mode]
    
    return

####################################################################
#########               EOT                             ############
#########               Accuracy                        ############
#########                                               ############
####################################################################

#-1 means that there is no trp in the episode
def eval_seq_pred(first_detect,comp_time='ts',threshold=0.74):
    out = []
    t_eval_dict = {i:[] for i in range(3)} 
    #print t_eval_dict
    for k,v in first_detect.iteritems():
        #print k,'\n',v
        if float(v['tg']) == -1:
            out.append((k,v))
            continue
        elif float(v[comp_time]) == -1:
            diff = '{:.2f}'.format(float(v[comp_time])-float(v['tg']))
            t_eval_dict[2].append(abs(float(diff)))
            continue
        else:
            if float(v[comp_time]) < float(v['tg'])+threshold:
                diff = '{:.2f}'.format((float(v['tg'])+threshold)-float(v[comp_time]))
                t_eval_dict[0].append(abs(float(diff)))    
                #if abs(float(diff))>0.02:
                #    print 'ts',v['ts']
                #    print 'tg',v['tg']
            elif float(v[comp_time]) == float(v['tg']):
                t_eval_dict[1].append('{:.2f}'.format(float(v['tg'])))
            else:
                diff = '{:.2f}'.format(float(v[comp_time])-(float(v['tg'])+threshold))
                t_eval_dict[2].append(abs(float(diff)))
                #print 'ts',v['ts']
                #print 'tg',v['tg']

    #print 'mean earlier:',np.mean([float(x) for x in t_eval_dict[0]])
    #print t_eval_dict[0]
    early_lt20 = len([float(x) for x in t_eval_dict[0] if float(x)<=0.76 and float(x)>0.0])
    #print 'number <= gold-0.02 to gold + 0.74', early_lt20
    early_gt20 = len([float(x) for x in t_eval_dict[0] if float(x)>0.76 or float(x)<0.0])
    #print 'number > gold + 0.74', early_gt20
    #print 'number predicted right:',len(t_eval_dict[1])
    #print 'mean later:',np.mean([float(x) for x in t_eval_dict[2]])
    #late_lt20 = len([float(x) for x in t_eval_dict[2] if float(x)<=0.75])
    #print 'number <= 0.75', late_lt20
    late_gt20 = len([float(x) for x in t_eval_dict[2]])
    #print 'number >= 0.75', late_gt20
    #print 'not in evaluation:',out 
    #print t_eval_dict[2]
    return t_eval_dict, early_lt20, early_gt20, late_gt20,len(out) 

#returns a dictionary {speaker:{prec_ts:Acc,prec_tp:Acc,cutin_ts:Acc,cutin_tp}}
def get_t_precision_dict(start=2, end=8, mode='lm',subset='test'):
    time_precision = {}
    for sp in range(start, end): 
        print sp
        ep_seq_dict = load_inc_dict(prepath,int(sp),int(sp)+1, mode, subset)
        ep_dict = deepcopy(ep_seq_dict[float(sp)])
        #print ep_seq_dict.keys()
        fname = './../../Logistic_Regression/r'+str(sp)+'/distributions/prob_dist_r'+str(sp)+'_'+mode+'_'+subset+'.pkl'
        ep_path = './../../pickledEpisodes/r'+str(sp)+'/r'+str(sp)+'_'
        pdist_df = load_pdist_df(fname, ep_path)
        
        #ep_seq_dict = ep_dict
        first_detect = get_fd_dict(ep_seq_dict, pdist_df)
        #print len(first_detect.keys())
        ############################
        t_eval_dict_ts, early_lt20_ts, early_gt20_ts, late_gt20_ts, out_ts = eval_seq_pred(first_detect,\
                                                                                                        comp_time='ts')
        num_eps_ts = float(len(first_detect.keys()))-out_ts
        prec_ts = early_lt20_ts/num_eps_ts
        ci_ts = early_gt20_ts/num_eps_ts
        #print 'ts', early_lt20_ts, num_eps_ts, prec_ts
        #################################################
        t_eval_dict, early_lt20, early_gt20, late_gt20, out = eval_seq_pred(first_detect,\
                                                                                      comp_time='tp')
        num_eps = float(len(first_detect.keys()))-out
        prec_tp = early_lt20/num_eps
        ci_tp = early_gt20/num_eps
        #print 'tp', early_lt20, num_eps, prec_tp
        #################################################
        time_precision[sp] = {'ts':prec_ts,'tp':prec_tp,'ts_cutin':ci_ts,'tp_cutin':ci_tp}

    return time_precision

def avg_EOTacc(time_precision_dict, start=2, end=8):
    TSvals = [time_precision_dict[speaker]['ts']\
              for speaker in range(start, end)] 
    TPvals = [time_precision_dict[speaker]['tp']\
              for speaker in range(start, end)] 
    avgTS = np.mean(TSvals)
    avgTP = np.mean(TPvals)
    return avgTS, avgTP

####################################################################
#########                                               ############
#########               Correction                      ############
#########                 Time                          ############
####################################################################

#get a dict with speaker:ep:(timetrpisdetected,correctiontime,finalOrnot)
def get_correction_time(models_path, start, end, mode, subset):

    print 'building dictionary with correction times...'
    
    correction_dict = {}
    for sp in range(start, end):
        print sp
        ep_seq_dict = load_inc_dict(models_path,int(sp),int(sp)+1, mode, subset)
        ep_dict = deepcopy(ep_seq_dict[float(sp)])
        
        fname = './../../Logistic_Regression/r'+str(sp)+'/distributions/prob_dist_r'+str(sp)+'_'+mode+'_'+subset+'.pkl'
        ep_path = './../../pickledEpisodes/r'+str(sp)+'/r'+str(sp)+'_'
        pdist_df = load_pdist_df(fname, ep_path)
        
        correction_dict[sp] = {}
                
        for ep in ep_dict: 
            correction_dict[sp][ep] = []
            inc_add = []
            detect_time = 0.0
            trp_det = False    
            for i in range(len(ep_dict[ep])):
                if trp_det == False\
                and 'trp10' in ep_dict[ep][i]:
                    trp_det = True
                    inc_add.append('trp10')
                    detect_time = (i+1)*0.01 
                    
                elif trp_det == True:
                    if isinstance(ep_dict[ep][i],list):
                        inc_add = inc_add[:-(len(ep_dict[ep][i])+1)]
                        inc_add += ep_dict[ep][i]
                    elif isinstance(ep_dict[ep][i],str):
                        inc_add.append(ep_dict[ep][i])
                    elif 'trp10' not in inc_add:
                        corr = ((i+1)*0.01)-detect_time
                        correction_dict[sp][ep].append((detect_time,corr,1))
                        inc_add = []
                        trp_det = False
                    
                    elif i == len(ep_dict[ep])-1:                        
                        corr = ((i+1)*0.01)-detect_time
                        correction_dict[sp][ep].append((detect_time,corr,2))
                        inc_add = []
                        trp_det = False
                else:
                    continue
                    
    return correction_dict

#In how many episodes TRP is predicted correct at first attempt?
#In how many episodes TRP is predicted correct after cutting in (not important how often)?
#In how many episodes TRP is not predicted correct (unimportant how often the system is cutting in)?
def real_corrections(prepath='./../../MMT/', start=2, end=8, mode='lm', subset='test'):
    
    real_corr_dict = {}
    ncorr = {}
    tpcorr = {}
    
    corr_dict = get_correction_time(prepath, start, end, mode, subset)        
    
    for sp in range(start, end): 
        print sp
        real_corr_dict[sp] = {}
        ncorr[sp] = [] 
        tpcorr[sp] = []
        
        ep_dict = load_inc_dict(prepath,int(sp),int(sp)+1, mode, subset)
        ep_seq_dict = deepcopy(ep_dict[float(sp)])
         
        fname = './../../Logistic_Regression/r'+str(sp)+'/distributions/prob_dist_r'+str(sp)+'_'+mode+'_'+subset+'.pkl'
        ep_path = './../../pickledEpisodes/r'+str(sp)+'/r'+str(sp)+'_'
        pdist_df = load_pdist_df(fname, ep_path)
        
        first_detect = get_fd_dict(ep_seq_dict, pdist_df)
        
        for k,v in first_detect.iteritems():
            if float(v['tg']) == -1:
                #out.append((k,v))
                continue
            elif corr_dict[sp][k] != []:
                if float(corr_dict[sp][k][-1][0]) <= float(v['tg']) \
                and float(corr_dict[sp][k][-1][0])+float(corr_dict[sp][k][-1][1]) >= float(v['tg']) :
                    if len(corr_dict[sp][k]) == 1:
                        tpcorr[sp].append(k)
                    else:
                        real_corr_dict[sp][k] = corr_dict[sp][k]
                else:
                    ncorr[sp].append(k)
            else:
                ncorr[sp].append(k)
                #print sp, k
                
    
    return real_corr_dict, ncorr, tpcorr

def print_corrections_per_threshold(real_corr_dict, ncorr, tpcorr, feat, start, end):
    thlist = [0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75]
    tlist, not_corr, msd = percent_threshold(real_corr_dict, thlist, start, end)

    print '\n Percentage of corrected TRP with various thresholds,\n mean and standard deviation of correction time in sec\n'
    print 'feature set: ', feat

    outstring = 'speaker\t 0.0\t '    
    outstring += '\t '.join([str(th) for th in thlist])
    outstring += '\t mean\t sd'
    outstring += '\n'

    th_dict = {th:[] for th in thlist}
    th_dict[0] = []
    
    for sp in range(start, end):
        #print sp
        
        
        outstring += str(sp)
                
        ksumn = len(ncorr[sp])
        ksumt = len(tpcorr[sp])
        ksumr = len(real_corr_dict[sp].keys())

        ges = ksumn+ksumt+ksumr 
        fcor = float(ksumn)/ges
        tcorr = float(ksumt)/ges
        rcorr =  float(ksumr)/ges

        outstring += '\t '+'{:.2f}'.format(tcorr)
        th_dict[0].append(tcorr)
        
        
        #print '\n',sp,'\n'

        #print fcor
        #print tcorr
        #print rcorr,'\n'

        for th, perc in tlist[sp]:
            #print th, (tcorr+(rcorr*perc))
            
            th_dict[th].append(tcorr+(rcorr*perc))
            
            outstring += ' \t '+'{:.2f}'.format(tcorr+(rcorr*perc))

        outstring += '\t '+'{:.2f}'.format(msd[sp]['mean'])
        outstring += '\t '+'{:.2f}'.format(msd[sp]['sd'])    
        outstring += '\n'

    outstring += 'avg'
    for th in sorted(th_dict):
        outstring += '\t '+'{:.2f}'.format(np.mean(th_dict[th]))
    outstring += '\n sd'
    for th in sorted(th_dict):
        outstring += '\t '+'{:.2f}'.format(np.array(th_dict[th]).std())
    outstring += '\n'
    
    return outstring

#return: dict with {speaker:{threshold: percentage of corrected TRP}}
def percent_threshold(corr_dict, thlist, start, end):
    cva = 0
    corr_times = []
    tlist_dict = {}
    msd = {}
    for sp in range(start, end):
        print sp
        msd[sp] = {'sd':0,'mean':0}
        for ep in corr_dict[sp]:
            if corr_dict[sp][ep] == []:
                #print ep
                continue
            else:
                for detect,correct,final in corr_dict[sp][ep]:
                    #if ep == 2 and sp == 2:
                    #    print detect, correct
                    if final != 2 and correct!=0:
                        corr_times.append(correct)
                    elif correct == 0:
                        cva+=1
                    else:
                        continue

        not_corrected = cva
        #print 'no correction of',cva,'FAs'

        time_list = ['{:.2f}'.format(t) for t in corr_times]
        time_list = sorted([float(t) for t in time_list])

        sd = np.array(time_list).std()
        mean_time = np.mean(time_list)
        msd[sp]['sd'] = sd
        msd[sp]['mean'] = mean_time
        #print 'standard deviation for correction time:\n', np.array(time_list).std()
        #print 'mean of correction time:\n', np.mean(time_list)

        tlist = [(th, len([t for t in time_list if t<=th])/float(len(time_list))) for th in thlist]
        
        tlist_dict[sp] = tlist
    return tlist_dict, not_corrected, msd

##### Plots
def plot_correction_delay(corr_dict, start=2, end=8, xmin=0.0, xmax=15.0, bin_tl=100, plegend=True):
    cva = 0
    corr_times = []
    for sp in range(start, end):
        print sp
        for ep in corr_dict[sp]:
            if corr_dict[sp][ep] == []:
                #print ep
                continue
            else:
                for detect,correct,final in corr_dict[sp][ep]:
                    #if ep == 2 and sp == 2:
                        #print detect, correct
                    if final != 2 and correct!=0:
                        corr_times.append(correct)
                    elif correct == 0:
                        cva+=1
                    else:
                        continue
    
    print 'no correction of',cva,'FAs'
    
    time_list = ['{:.2f}'.format(t) for t in corr_times]
    time_list = sorted([float(t) for t in time_list])

    #print min(time_list)
    
    mean_corr = np.mean(time_list)
    std_corr = np.array(time_list).std()
    print 'mean',mean_corr,'sd',std_corr

    ymax = 100
    
    #cumulative = np.cumsum(time_list)
    #plt.plot(cumulative)

    plt.hist(time_list, bins=bin_tl, histtype='step', cumulative=True, weights=[ymax/float(len(time_list))]*len(time_list), alpha=0.5, color='r', label='cumulative frequency')
    plt.hist(time_list, bins=bin_tl, histtype='stepfilled', cumulative=False, weights=[ymax/float(len(time_list))]*len(time_list), alpha=0.5, color='b', label='correction time')
    
    
    plt.plot([mean_corr-std_corr, mean_corr-std_corr], [0.0, ymax], color='b', linestyle='--', linewidth=.5)
    plt.plot([mean_corr+std_corr, mean_corr+std_corr], [0.0, ymax], color='b', linestyle='--', linewidth=0.5)

    plt.plot([0.75, 0.75], [0.0, ymax], color='r', linestyle='--', linewidth=.5)

    #plt.plot([0.1, 0.1], [0.0, ymax], color='r', linestyle='--', linewidth=.5)
    #plt.plot([0.0, 15.0], [76, 76], color='b', linestyle='--', linewidth=.5)

    
    plt.ylim(0,ymax)
    plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
    xti = [numb*0.01 for numb in [0,10,20,30,40,50,60,70,80,90,100]]
    plt.xticks(xti)
    plt.xlim(xmin,xmax)
    #plt.xlim(-2.0,15.)

    plt.title("Distribution of correction time\n and cumulative frequency")
    plt.xlabel("correction time in sec")
    plt.ylabel("frequency")
    if plegend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    return

####################################################################
#########                                               ############
#########               Edit                            ############
#########               Overhead                        ############
####################################################################

def compute_edit_overhead(prepath='./../../MMT/', start=2, end=8, mode='lm', subset='test'):
    
    edit_overhead = {}
    
    for speak in range(start, end):
        print speak
        edit_overhead[speak] = {}
        
        ep_seq_dict = load_inc_dict(prepath, int(speak),int(speak)+1, mode, subset)
        ep_dict = deepcopy(ep_seq_dict[float(speak)])
        
        #t_dist_dict_ts = get_temporal_diff_distribution(comp_time = 'ts', mode='lm',subset='dev')
        
        #time_list_ts = ['{:.2f}'.format(t) for speak in t_dist_dict_ts for t in t_dist_dict_ts[speak]]
        #time_list_ts = sorted([float(t) for t in time_list_ts]) 
        #mean_ts = np.mean(time_list_ts)
        
        #edit_overhead[speak]['avg'] = mean_ts
        operations = 0
        out = []
        
        for ep in ep_dict: 
            inc_seq = []            
            for i in range(len(ep_dict[ep])):                
                if isinstance(ep_dict[ep][i],list):
                    operations += 1                    
                    for inc in inc_seq[-(len(ep_dict[ep][i])+1):-1]:
                        for inc_sub in ep_dict[ep][i]:
                            if inc != inc_sub:
                                operations += 1
                    inc_seq = inc_seq[:-(len(ep_dict[ep][i])+1)]                    
                    inc_seq += ep_dict[ep][i]                    
                elif isinstance(ep_dict[ep][i],str):
                    operations += 1
                    inc_seq.append(ep_dict[ep][i])
            try:
                eo = float(operations)/len(ep_dict[ep])
                
                if eo >200:
                    out.append((speak,ep,eo))
                edit_overhead[speak][ep] = float('{:.2f}'.format(eo))
            except:
                print 'Error in:'
                print speak,ep
                print ep_dict[ep]
    print 'out:',out    
    return edit_overhead

def print_eo(eo_dict, start=2, end=8):
    print 'Mean and standard deviation of Edit Overhead per speaker:'
    for sp in range(start, end):
        eo_per_speaker = [eo_dict[sp][ep] for ep in eo_dict[sp]]
        mean = np.mean(eo_per_speaker)
        sd = np.array(eo_per_speaker).std()
        print 'speaker:',sp
        print 'mean: ', '{:.2f}'.format(mean)
        print 'standard deviation: ','{:.2f}'.format(sd),'\n'
                    
    return

def plot_eo_distribution(speaker, eo_dict, title, plegend=True, bins_tp=60, for_all=False):
    if for_all:
        time_list = [float('{:.2f}'.format(eo_dict[speak][ep])) for speak in eo_dict for ep in eo_dict[speak]]
        print 'times all speakers', len(time_list)
        #print time_list
    else:
        time_list = [float('{:.2f}'.format(eo_dict[speaker][ep])) for ep in eo_dict[speaker]]
        print 'times speaker', speaker, len(time_list)
        #print time_list
        #print time_list_ts
    
   
    mean_corr = np.mean(time_list)
    std_corr = np.array(time_list).std()
    print 'mean',mean_corr,'std',std_corr
    
    ymax = 100
    
    #time_list = sorted([float(t) for t in time_list])
    #if len(time_list) != 0:
    plt.hist(time_list, bins=bins_tp, histtype='step', cumulative=True, weights=[float(ymax)/len(time_list)]*len(time_list), alpha=0.5, color='r', label='cumulative frequency')
    plt.hist(time_list, bins=bins_tp, histtype='stepfilled', weights=[float(ymax)/len(time_list)]*len(time_list), alpha=0.5, color='b')#, label='incremental')
    
    
    plt.plot([mean_corr-std_corr, mean_corr-std_corr], [0.0, ymax], color='b', linestyle='--', linewidth=.5)
    plt.plot([mean_corr+std_corr, mean_corr+std_corr], [0.0, ymax], color='b', linestyle='--', linewidth=0.5)
    
    plt.ylim(0,ymax)
    plt.xlim(-50.0,200)
    if title != '':
        plt.title(title)
    elif for_all:
        plt.title("Histogram of edit overhead all speakers")
    else:
        plt.title("Histogram of edit overhead speaker "+str(speaker))
    plt.xlabel("edit overhead")
    plt.ylabel("Frequency")
    if plegend:
        plt.legend()
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    return
