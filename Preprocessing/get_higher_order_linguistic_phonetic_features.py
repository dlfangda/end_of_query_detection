#!python

#from os import walk, path, makedirs 
import pandas as pd
import numpy as np
import pickle
import tgt
from shutil import copy
import os
import codecs

from extract_utils import *
from LM_utils import *

def get_higher_order_LMfeats_zscore(ep_dir = "./pickledEpisodes",\
                                    tg_dir='./../../../Masterarbeit/Programmierung/take_the_turn/tg_eps/',\
                                    out="./pickledEpisodes_higherOrderTextGridData",\
                                    utt_path = "./utterances_for_LM.txt", debug=False, test_paths_list=[]):
    
    #First get paths to all episodes used
    if test_paths_list != []: #if testing on a subset
        ep_paths = ['r'+str(tupel[1])+'_'+str(tupel[2])+'.pkl' for tupel in test_paths_list]
    else:
        ep_paths = sorted(lsdir(ep_dir,'.pkl')) #raw path to all pkl files
    
    #print ep_paths
    err_list = []
    
    if not os.path.isfile(utt_path):
        #Get textgrid files
        tgfiles = lsdir(tg_dir, ".TextGrid")
        #if needed to be created, create text with utterances for LM
        utttxt = codecs.open(utt_path,"w","utf8")
    
        last_speaker = 1
        progress_count = 0
        for eps_path in tgfiles:
            speaker = eps_path.split('_')[0][1]
            ep = int(eps_path.split('_')[1].split('.')[0])

            #print speaker,ep
            if int(speaker) == 1:
                continue
                
            if speaker != last_speaker:
                print speaker
                last_speaker = speaker                
                progress_count = 0
                
            if progress_count!=0 and progress_count%10==0:
                print 'progress',progress_count

            ep_df = pd.DataFrame()
            
            tg_path = tg_dir +"r"+str(speaker)+"/r"+str(speaker)+"_"+str(int(ep))+".TextGrid"
            
            try:
                if debug == False:
                    utttxt.write(str(int(speaker))+" "+str(int(ep))+" ")
                utt = utt_for_file(tg_path, utttxt)
                if debug == False:
                    utttxt.write(utt)
                else:
                    print "utterance of ",speaker, ep, utt
            except:
                err_list.append(('NoneTypeError',speaker,ep))
                continue
            progress_count += 1
        utttxt.close()
        
    ######################################
    ############bulit lm##################
    ######################################    
    if not debug:
        if not os.path.isdir(out):
            os.makedirs(out)

        for sp in range(2,8):
            speaker_ep_dir = out+'/r'+str(sp)
            if not os.path.isdir(speaker_ep_dir):
                os.makedirs(speaker_ep_dir)
    
    progress_count = 0
    eps_dict = get_eps_from_pathslist(ep_paths)
    
    for speaker in eps_dict.keys():
        print "speaker", speaker

        if debug:
            speaker = dspeaker
            ep = dep
            
        progress_count = 0
        print "build lm excluding speaker", speaker
        lm = build_lm(utt_path, eps_dict, speaker, 2, False)
        dur_list_for_speaker = get_durations_for_speaker(tg_dir, int(speaker), eps_dict[int(speaker)])
        #build dictionary for zscore
        counter = 0
        for ep in eps_dict[speaker]:
            path_ep = ep_dir+'/r'+str(speaker)+'/r{0}_{1}.pkl'.format(speaker,ep) 
           
            #print path_ep
            ep_df = pickle.load(open(path_ep,"rb"))
            gold_word_list = ep_df.gold_words.tolist()
            gold_cword_list = ep_df.gold_current_word.tolist()
            wordlist = ep_df.words.tolist()
            cwordlist = ep_df.current_word.tolist()
            
            #function aus LM_utils
            #exclude = last_speaker, predict for the current episode!
            ldur_list, label_list, wml_list, wml_trigram_list, entropy_list = \
                       apply_lm(lm,gold_cword_list, gold_word_list, wordlist, cwordlist, nspeaker=8, n=2, mtp=False)
            #break
            if wordlist == [] or wml_list == []:
                #print [a.text.encode('utf8') for a in word_list], [a.encode('utf8') for a in wordlist], wml_list
                err_list.append(('valueError',speaker,ep))
                continue

            #zscores:
            zsc_list = get_zsc_per_ep(ep_df, dur_list_for_speaker[int(speaker)])

            if ldur_list[-1] < 15.00:
                #print len(ep_df)
                tail_ep = ep_df.tail(1)
                av = int((15.00-ldur_list[-1])*100)
                #print av
                for i in range(av):
                    add_rows = pd.DataFrame({'speaker':speaker,\
                                  'episode':ep, 'time_in_sec':(len(ep_df)*0.01)+(i+1),\
                                  'phones': tail_ep.phones.values[0].encode('utf8'),\
                                  'duration':tail_ep.duration.values[0],\
                                  'words': tail_ep.words.values[0].encode('utf8'),\
                                  'current_word': tail_ep.current_word.values[0].encode('utf8'),\
                                  'gold_words': tail_ep.gold_words.values[0].encode('utf8'),\
                                  'gold_current_word': tail_ep.gold_current_word.values[0].encode('utf8')},\
                                 index = [(len(ep_df)-1)+(i+1)])
                    ep_df = pd.concat([ep_df, add_rows])

                    ldur_list.append(ldur_list[-1]+((i+1)*0.01))
                    label_list.append(label_list[-1])
                    wml_list.append(wml_list[-1])
                    wml_trigram_list.append(wml_trigram_list[-1])
                    entropy_list.append(entropy_list[-1])
                    zsc_list.append(zsc_list[-1])

            #print len(ep_df)
            #break
            try:
                ep_df['wml'] = np.array(wml_list).astype(float)
                ep_df['wml_trigram'] = np.array(wml_trigram_list).astype(float)
                ep_df['entropy'] = np.array(entropy_list).astype(float)
                ep_df['label'] = np.array(label_list).astype(float)
                ep_df['label_dur'] = np.array(ldur_list).astype(float)
                ep_df['zscore'] = np.array(zsc_list).astype(float)
            except:
                print len(ep_df), len(zsc_list),len(wml_list),\
                      wml_list[-1], ldur_list[-1], label_list[-1]
            #    err_list.append(('lengthError',speaker,ep))
            #    continue

            #new_index = [i for i in range(len(ep_df))]
            #ep_df.reindex(new_index)

            ep_out = out+'/r'+str(speaker)+'/r'+str(speaker)+"_"+str(int(ep))+".pkl"
            if len(ep_df)<1500:
                print speaker, ep
                err_list.append(('labeldurError',speaker,ep))
            else:       
                pkl_dataframe(ep_df, ep_out)
                #print ep_out
                #return ep_df

            progress_count += 1
            counter += 1
            if progress_count!=0 and progress_count%10==0:
                print 'progress',progress_count



    print err_list
    return

if __name__ == '__main__':    
    text_grid_dir = "../../../eot_detection_data/Data/textgrids"
    #using separate higherorder data position before merging
    data_dir = "../../../eot_detection_data/Data/pickled_episodes_ling_features"
    out_dir = data_dir
    try:
        os.mkdir(out_dir)
    except OSError:
        print "couldn't make dir",out_dir,"or already there."

    get_higher_order_LMfeats_zscore(ep_dir = data_dir,\
                                    tg_dir = text_grid_dir,\
                                    out = out_dir,\
                                    utt_path = "./utterances_for_LM.txt", debug=False, 
                                    test_paths_list=[])