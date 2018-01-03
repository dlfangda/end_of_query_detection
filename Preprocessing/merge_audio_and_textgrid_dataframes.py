#!python

import os
import pandas as pd
import numpy as np

import sys  
stdout = sys.stdout

reload(sys)  
sys.setdefaultencoding('utf8')
sys.getdefaultencoding()

sys.stdout = stdout

import extract_utils as eu

def adjust_col_length(ep_len, adj_col):
    #has to be adjusted for columns with rms_minus_* values
    #here one can use the values from the original rms column
    #or past values of the current column
    if len(adj_col) < ep_len:
        av = ep_len - len(adj_col)
        adj_col += [adj_col[-1]]*av
    elif len(adj_col) > ep_len:
        adj_col = adj_col[:ep_len]    
    return adj_col

#main function:
def merge_dataframes(ep_path_tgData="./pklEpisodes_tgData",\
                  ep_path_wavData="./pklEpisodes_wavData",\
                  out="./pklEpisodes",\
                  paths_list = []):

    if not os.path.isdir(out):
        os.makedirs(out)

    for sp in range(2,8):
        ep_dir = out+'/r'+str(sp)
        if not os.path.isdir(ep_dir):
            os.makedirs(ep_dir)
    
    if paths_list != []:
        ep_paths = ['r'+str(tupel[0])+'_'+str(tupel[1])+'.pkl' for tupel in paths_list]
    else:
        ep_paths = eu.lsdir(ep_path_tgData,'.pkl')
        ep_paths = sorted(ep_paths)
    
    print len(ep_paths)
    wav_paths = eu.lsdir(ep_path_wavData, '.pkl')
    print len(wav_paths)

    set_wav_out_paths = set(ep_paths) - set(wav_paths)
    print "wav paths not in loop: ",set_wav_out_paths
    
    set_textgrid_out_paths = set(wav_paths) - set(ep_paths)
    print "textgrid paths not in loop: ", set_textgrid_out_paths
    
    last_speaker = 1
    progress_count = 0
    err_list = []
    for eps_path in ep_paths:
        
        speaker = eps_path.split('_')[0][1]
        ep = int(eps_path.split('_')[1].split('.')[0])

        #print speaker,ep
        if int(speaker) == 1:
            continue
            
        if speaker != last_speaker:
            print speaker
            last_speaker = speaker                
            progress_count = 0
            print set(err_list)
            err_list = []
            
        if progress_count!=0 and progress_count%10==0:
            print 'progress',progress_count

        eppath_tg = ep_path_tgData+"/r"+str(speaker)+"/"
        eppath_wav = ep_path_wavData+"/r"+str(speaker)+"/"
        tg_df = eu.open_pkl_ep(speaker, ep, eppath_tg)
        wav_df = eu.open_pkl_ep(speaker, ep, eppath_wav)

        #print wav_df.keys()
        wav_df.drop('frameTime',1,inplace=True)
        
        new_ep_df = tg_df.copy()
        new_ep_df = new_ep_df.drop(new_ep_df[(new_ep_df.label == 2)&(new_ep_df.label_dur>13.00)].index)

        current_labels = new_ep_df.label.tolist()
        
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
            print trp_start, "trp number: ", current_labels.count(2)
            continue 

        len_minus_one = False
        equal_length = True
        ep_len = len(new_ep_df)
        if ep_len != len(wav_df):
            equal_length = False
            if ep_len - len(wav_df) < 1 and ep_len - len(wav_df) > 0 :
                len_minus_one = True
            #elif ep_len - len(wav_df) > 2:
            #    err_list.append((speaker,ep,len(adj_col), ep_len,'wav shorter'))
        
        if len_minus_one == True:
            print speaker, ep, ep_len,len(wav_df)
            new_ep_df = wav_df.copy()
            av = ep_len - len(wav_df)
            for key in tg_df.keys():
                new_ep_df[key] = np.array(tg_df[key].tolist()[:-av])
        else:       
            for key in wav_df.keys():
                if equal_length == False:
                    adj_col = wav_df[key].tolist()
                    if len(adj_col) < ep_len:
                        #print 'wav file shorter then tg file'
                        #print len(adj_col), ep_len
                        err_list.append((speaker,ep,len(adj_col), ep_len,'wav shorter'))
                        continue
                    else:
                        adj_col = adjust_col_length(ep_len, adj_col)
                    new_ep_df[key] = np.array(adj_col)
                else:
                    new_ep_df[key] = np.array(wav_df[key].tolist())

        if not 'pcm_intensity_sma' in new_ep_df.keys():
            print 'not merged',speaker,ep
            continue
        else:    
            ep_out = out + "/r"+str(speaker)+"/r"+str(speaker)+"_"+str(ep)+".pkl"
            eu.pkl_dataframe(new_ep_df, ep_out)

    print "Errors- please check!", set(err_list)
    for f in set_wav_out_paths:
        speaker_folder = f[:2]
        print "deleting textgrid pkl with no corresponding wav", f
        os.remove(os.path.join(ep_path_tgData,speaker_folder,f))
    for f in set_textgrid_out_paths:
        speaker_folder = f[:2]
        print "deleting wav pkl with no corresponding textgrid", f
        os.remove(os.path.join(ep_path_wavData,speaker_folder,f))
    return 

#####################
# Now cut the dataframe at label 2 with label duration 13.0
# longest MTP 12.22sec
#####################

if __name__ == "__main__":
    #adjust pathes to your machine!
    #ep_path and out can be the same path
    #ep_path_tg_data_path = "./pickledEpisodes_higherOrderTextGridData"
    ep_path_tg_data = "../../../eot_detection_data/Data/pickled_episodes_ling_features"
    #ep_path_wav_data = "./../../eot_detection_data/pickled_episodes_1"
    ep_path_wav_data = "../../../eot_detection_data/Data/pickled_episodes_acoustic_features"
    #out_path  = "./pickled_episodes_old_wav13sec"
    out_path = "../../../eot_detection_data/Data/pickled_episodes"

    elist = merge_dataframes(ep_path_tgData=ep_path_tg_data,\
                      ep_path_wavData=ep_path_wav_data,\
                      out=out_path,\
                      paths_list = [])
