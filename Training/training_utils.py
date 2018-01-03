import pickle
import os

def round_down(num, divisor):
    return num - (num%divisor)

#args: rootdir -> root directory, ending -> file ending
#return: list of pathes in rootdir 
def lsdir(rootdir, ending):
    pathlist = []
    for root,  dirs,  files in os.walk(rootdir,  topdown=False):
        for filename in files:
            if ending in filename:
                pathlist.append(filename)
    return pathlist

#read in a pickled dataframe for an episode in TAKE
#input: speaker, episode number, directorypath
#return: unpickled dataframe for episode
def open_pkl_ep(speaker,ep,ep_path):
    fname = ep_path+'r'+str(int(speaker))+'_'+str(ep)+'.pkl'
    
    with open(fname,'rb') as fp:
        ep_df = pickle.load(fp)
    #ep_df = ep_df[['speaker', 'episode', 'time_in_sec', \
    #               'label','label_dur',\
    #               'pcm_LOGenergy_sma', 'pcm_RMSenergy_sma',\
    #               'pcm_intensity_sma', 'intensity_mean','intensity_slope', \
    #               'pcm_loudness_sma', \
    #               'phones',\
    #               'duration', 'zscore', \
    #               'wml',  'wml_trigram',\
    #               'rms_minus_four',  'rms_minus_one', \
    #               'rms_minus_three',  'rms_minus_two',\
    #               'voicingFinalUnclipped_sma', 'F0final_sma',
    #               'F0raw_sma', 'F0_mean','F0_slope', 'entropy' ]]
    return ep_df
