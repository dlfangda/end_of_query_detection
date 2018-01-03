#!python

import os
import pandas as pd
import numpy as np

import extract_utils as eu

def window_over_col(df,cname='pcm_RMSenergy_sma' ,wsize=4):
    windowed_cols_list = []

    #rms energy minus number
    rms_vals = df[cname].values.tolist()

    for i in range(wsize):
        rms_minus_vals = [0.0]*(i+1)
        rms_minus_vals += rms_vals[:-(i+1)]
        windowed_cols_list.append(rms_minus_vals)
        
    return windowed_cols_list


def get_mean_and_slope(df,cname,wsize):
    mlist = []
    slopelist = []
    slop = 0

    df_col = df[cname].values.tolist()
    
    for i in range(len(df_col),0,-1):
        try:
            wvals = df_col[i-wsize:i]
            if not len(wvals) == 0:
                intens_mean = eu.mean(wvals)
            else:
                intens_mean = eu.mean(df_col[0:i])
                wvals = df_col[0:i]
        except IndexError, ZeroDivisionError:
            intens_mean = eu.mean(df_col[0:i])
            wvals = df_col[0:i]
        mlist.append(intens_mean)
            
        for x in range(1,len(wvals)):
            slop += list(wvals)[x]-list(wvals)[x-1]
            
        slopelist.append(slop)        
        slop = 0

    #because I iterate from the end to the beginning of the dataframe
    #I have to reverse the lists of slope and mean
    slopelist = list(reversed(slopelist))
    mlist = list(reversed(mlist))
    
    return mlist, slopelist

#main function:
#args: column names, path to pkl episodes, out path,
#optional list of episodes [(speaker, ep)] for debugging purposes
#
#col_names: [(name of column, window size, prefix of new column, what is computed?)]
#two values for what is computed: mean:mean and slope of a specified window,
#window: window backwards over column)]
def loop_over_eps(col_names = [('pcm_intensity_sma',15,'intensity','mean'),\
                           ('pcm_RMSenergy_sma',4,'rms','window')],\
                  ep_path="./pklEpisodes_rawfeats",\
                  out="./pklEpisodes_higherfeats",\
                  paths_list = []):

    out_eps = []
    #out = [(3,203),(4,252),(6,195),(7,33),(7,82)]
    translate = {1:'one',2:'two',3:'three',4:'four',5:'five',6:'six'}

    if not os.path.isdir(out):
        os.makedirs(out)

    for sp in range(2,8):
        ep_dir = out+'/r'+str(sp)
        if not os.path.isdir(ep_dir):
            os.makedirs(ep_dir)
    
    if paths_list != []:
        ep_paths = ['r'+str(tupel[0])+'_'+str(tupel[1])+'.pkl' for tupel in paths_list]
    else:
        ep_paths = eu.lsdir(ep_path,'.pkl')
        ep_paths = sorted(ep_paths)
    

    last_speaker = 1
    progress_count = 0
    for eps_path in ep_paths:
        #print eps_path
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

        eppath = ep_path+"/r"+str(speaker)+"/"
        ep_df = eu.open_pkl_ep(speaker, ep, eppath)
        new_ep_df = ep_df.copy()
        #print new_ep_df.keys()
        #print new_ep_df.tail(1).label_dur
        #break
        #new_ep_df = new_ep_df[['speaker','episode', 'pcm_intensity_sma','pcm_RMSenergy_sma']]
        #print new_ep_df.keys()

        
        if (int(sp),int(ep)) not in out_eps:
            for cname in col_names:
            #another idea for a higher feature: wml slope, mean
            #if cname =='wml':
            #    mlist, slopelist = window_over_words(sub_df,cname)
                pre = cname[2]
                if "window" in cname[3]:
                    list_rms_vals = window_over_col(ep_df,cname[0] ,wsize=cname[1])
                    for i in range(len(list_rms_vals)):
                        new_ep_df[pre+'_minus_'+str(translate[i+1])] = np.array(list_rms_vals[i])
                elif "mean" in cname[3]:
                    mlist, slopelist = get_mean_and_slope(ep_df,cname[0],cname[1])
                    new_ep_df[pre+'_mean'] = np.array(mlist)
                    new_ep_df[pre+'_slope'] = np.array(slopelist)
                
        
        ep_out = out + "/r"+str(speaker)+"/r"+str(speaker)+"_"+str(ep)+".pkl"
        eu.pkl_dataframe(new_ep_df, ep_out)
        progress_count+=1

    return

if __name__ == "__main__":
    #adjust pathes to your machine!
    #ep_path and out can be the same path
    #adjust pathes to your machine!
    #ep_path and out can be the same path
    ep_path = "../../../eot_detection_data/Data/pickled_episodes_acoustic_features"
    out_dir = ep_path
    try:
        os.mkdir(out_dir)
    except OSError:
        print "couldn't make dir",out_dir,"or already there."

    loop_over_eps(col_names = [('pcm_intensity_sma',15,'intensity','mean'),\
                               ('F0final_sma',15,'F0','mean'),\
                               ('voicingFinalUnclipped_sma',15,'voicingFinalUnclipped','mean'),\
                               ('pcm_RMSenergy_sma',4,'rms','window')],\
                      ep_path=ep_path,\
                      out=out_dir,\
                      paths_list = [])
