# coding: utf-8

"""Script to extract audio features 
from wav files for each episode using opensmile
(which must be installed).
It adds these features to the existing pickled
dataframes with features and overwrites them
"""
import os
import shutil
import argparse
import pandas as pd

import extract_utils as eu


def extract_raw_audio_features_from_wav(rootwavdir,rootcsvdir,rootpkldir,opensmile):
    
    #1. get the opensmile features and write to temp csv folder
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
    except:
        print "couldn't get folder of this file, use local absolute"
        dir_path = "/Users/julianhough/git/eot_detection/Code/Preprocessing"

    opensmile_config = dir_path + "/combi_prosody.conf"
    try:
        os.mkdir(rootcsvdir)
    except OSError:
        print "already made root csv folder",rootcsvdir

    for wavdir in sorted(os.listdir(rootwavdir)):
        abswavdir = os.path.join(rootwavdir,wavdir)
        if not os.path.isdir(abswavdir): 
            continue
        print "speaker",wavdir
        if "1" in wavdir: continue #omitting r1
        wavfiles = os.listdir(abswavdir)
        csv_speaker_folder = os.path.join(rootcsvdir,wavdir)
        try:
            os.mkdir(csv_speaker_folder)  
        except OSError:
            print "already made speaker csv folder", csv_speaker_folder
        for wav in sorted(wavfiles):
            wavfile = os.path.join(rootwavdir,wavdir,wav)
            csv = os.path.join(csv_speaker_folder,wav.replace(".wav",".csv"))
            command = opensmile + ' -nologfile -C {} -I "{}" -O "{}"'\
                    .format(opensmile_config,wavfile,csv)
            #print command
            print wavfile
            os.system(command)
            #h = raw_input()
            #if h == "q": break
        #h = raw_input()
        #if h == "q": break

    #2. transfer the temp csv file data to the existing
    #pickles and overwrite those pickled dataframes
    try:
        os.mkdir(rootpkldir)
    except OSError:
        print "already made root pkl folder",rootpkldir


    for csvdir in sorted(os.listdir(rootcsvdir)):
        csv_speaker_folder = os.path.join(rootcsvdir,csvdir)
        if not os.path.isdir(csv_speaker_folder): 
            continue
        print "speaker",csvdir
        if "1" in csvdir: continue #ommitting r1
        csvfiles = os.listdir(csv_speaker_folder)
        #csv_speaker_folder = abswavdir.replace("/wav_eps","/take_csv_eps")
        pkl_speaker_folder = os.path.join(rootpkldir,csvdir)
        try:
            os.mkdir(pkl_speaker_folder)  
        except OSError:
            print "already made speaker pkl folder", pkl_speaker_folder
        for csv in sorted(csvfiles):
            print csv
            if not ".csv" in csv: continue
            csvfile = os.path.join(rootcsvdir,csvdir,csv)
            print csvfile
            df_pkl_file = os.path.join(rootpkldir,csvdir,
                                       csv.replace('.csv',".pkl"))
            #same folder or not?
            df = pd.DataFrame.from_csv(csvfile,sep=";")
            df.frameTime = df.frameTime + 0.01

            eu.pkl_dataframe(df, df_pkl_file)
    shutil.rmtree(rootcsvdir) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opensmile', help="Location of OpenSmile", default = \
                        "/Applications/opensmile-2.0-rc1/opensmile/inst/bin/SMILExtract")
    parser.add_argument('-audio', help="Location of audio data", default = \
                        "/Users/julianhough/sciebo/Angelika_EOT/wav_eps")
    args = parser.parse_args()
    #the wav files
    #where the temporary csv output from OpenSmile should go:
    rootcsvdir = "../../../eot_detection_data/Data/pickled_episodes_temp_csv"
    #target data dir for the pickles
    rootpkldir = "../../../eot_detection_data/Data/pickled_episodes_acoustic_features"

    extract_raw_audio_features_from_wav(args.audio,rootcsvdir,rootpkldir,args.opensmile)