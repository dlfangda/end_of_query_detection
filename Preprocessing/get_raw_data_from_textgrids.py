#!python

import os
import pandas as pd
import numpy as np
import pickle
import tgt

import extract_utils as eu

def repair_length(col, ep_length):
    if len(col) < ep_length:
        av = ep_length - len(col)
        col += [col[-1]]*av
    else:
        if len(col) > ep_length:
            av = len(col) - ep_length
            col = col[:-av]
    return col

def get_data_from_textgrids(tg_path='./../../../Masterarbeit/Programmierung/take_the_turn/tg_eps',\
                            out='./pickledEpisodes', paths_list = []):
    #look into the episodes why they are left out
    #out = [(3,203),(4,252),(6,195),(7,33),(7,82)]

    #print tg_path

    if not os.path.isdir(out):
        os.makedirs(out)

    for sp in range(2,8):
        ep_dir = out+'/r'+str(sp)
        if not os.path.isdir(ep_dir):
            os.makedirs(ep_dir)
    
    if paths_list != []:
        ep_paths = ['r'+str(tupel[0])+'_'+str(tupel[1])+'.pkl' for tupel in paths_list]
    else:
        ep_paths = eu.lsdir(tg_path,'.TextGrid')
        ep_paths = sorted(ep_paths)

    #print ep_paths

    last_speaker = 1
    progress_count = 0
    err_list = []
    for eps_path in ep_paths:
        speaker = int(eps_path.split('_')[0][1])
        ep = int(eps_path.split('_')[1].split('.')[0])

        #print speaker,ep
        if speaker == 1:
            continue
            
        if speaker != last_speaker:
            print speaker
            last_speaker = speaker                
            progress_count = 0
            
        if progress_count!=0 and progress_count%10==0:
            print 'progress',progress_count


        ep_df = pd.DataFrame()
        tgpath = tg_path +"/r"+str(speaker)+"/"+ "r"+str(speaker)+"_"+str(ep)+".TextGrid"

        
        #get raw durations as list of values
        #with length end_time last_word + 15sec
        #try:
        #print tgpath
        dur_list, phone_list, ep_length = eu.get_durations_for_ep(tgpath)
        #except:
        #    err_list.append((speaker,ep))
        #    continue

        word_list, cword_list = eu.get_words(tgpath)

        #print speaker, ep

        time_list, sp_list, ep_list,\
                   gold_cword_list, gold_word_list, wordlist,\
                   cwordlist = eu.get_word_cols(word_list,\
                                            cword_list,\
                                            ep_length, speaker, ep)

        columns = [time_list, sp_list, ep_list,\
                   gold_cword_list, gold_word_list, wordlist,\
                   cwordlist]
        
        if len(gold_cword_list) != ep_length:
            gold_cword_list = repair_length(gold_cword_list, ep_length)
        if len(gold_word_list) != ep_length:
            gold_cword_list = repair_length(gold_word_list, ep_length)
        if len(wordlist) != ep_length:
            wordlist = repair_length(wordlist, ep_length)
        if len(cwordlist) != ep_length:
            cwordlist = repair_length(cwordlist, ep_length)
        
        try:
            ep_df['speaker'] = np.array(sp_list).astype(float)
            ep_df['episode'] = np.array(ep_list).astype(float)
            ep_df['time_in_sec'] = np.array(time_list).astype(float)
            ep_df['phones'] = np.array(phone_list)
            ep_df['duration'] = np.array(dur_list).astype(float)
            ep_df['words'] = np.array(wordlist)
            ep_df['current_word'] = np.array(cwordlist)
            ep_df['gold_words'] = np.array(gold_cword_list)
            ep_df['gold_current_word'] = np.array(gold_word_list)
        except:
            err_list.append((speaker,ep))
            continue
            
        out_path = out+"/r"+str(speaker)+\
                 "/r"+str(speaker)+"_"+str(ep)+".pkl"
        eu.pkl_dataframe(ep_df, out_path)
        progress_count+=1

    print err_list
    return

if __name__ == '__main__':
    text_grid_dir = "../../../eot_detection_data/Data/textgrids"
    data_dir = "../../../eot_detection_data/Data/pickled_episodes_ling_features"
    try:
        os.mkdir(data_dir)
    except OSError:
        print "couldn't make dir",data_dir,"or already there."

    get_data_from_textgrids(tg_path=text_grid_dir,\
                                out=data_dir, paths_list = [])