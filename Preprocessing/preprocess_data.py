import os
import sys
import subprocess
import argparse

def main(args):
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
    except:
        print "couldn't get folder of this file, use local absolute"

    if args.audio:
        print "extracting raw audio features..."
        subprocess.call([sys.executable, dir_path + '/get_raw_data_from_audio_files.py', 
                         "-opensmile", args.opensmile, "-audio", args.audio])
        print "done extracting raw audio features"
    else:
        print "using already extracted raw audio features"
    
    print "computing higher order audio features..."
    subprocess.call([sys.executable, dir_path + '/get_higher_order_acoustic_features.py'])
    print "done computing higher order audio features"
    
    print "extracting raw lexical and phoneme duration features..."
    subprocess.call([sys.executable, dir_path + '/get_raw_data_from_textgrids.py'])
    print "done extracting raw lexical and phoneme duration features"
    
    print "computing higher order lexical and phoneme length features..."
    subprocess.call([sys.executable, dir_path + '/get_higher_order_linguistic_phonetic_features.py'])
    print "done computing higher order lexical and phoneme length features"
    
    print "merging all data together into pickles..."
    subprocess.call([sys.executable, dir_path + '/merge_audio_and_textgrid_dataframes.py'])
    print "done merging all data together into pickles"
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opensmile', help="Location of OpenSmile", default = \
                        "/Applications/opensmile-2.0-rc1/opensmile/inst/bin/SMILExtract")
    parser.add_argument('-audio', help="Location of audio data", default = \
                        "/Users/julianhough/sciebo/Angelika_EOT/wav_eps")
    args = parser.parse_args()
    main(args)