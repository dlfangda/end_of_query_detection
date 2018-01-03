#!python

import numpy
import pandas as pd
import sys
import os
import tgt
import codecs
from random import shuffle

from extract_utils import *
sys.path.append(os.path.join(os.getcwd(),".")) #path to the parent of the 'language_model' folder
from language_model.ngram_language_model import KneserNeySmoothingModel

#something goes wrong here - no print statements possible any longer when doing reload(sys)
#solution: save and redirect the standard output 
import sys  
stdout = sys.stdout

reload(sys)  
sys.setdefaultencoding('utf8')
sys.getdefaultencoding()

sys.stdout = stdout

def get_col(word_list, wav_len, gold=False):

    wordlist = []

    lw = ""
    last_mult = 0
    for i in range(len(word_list)):
        if gold == True:
            if i != len(word_list)-1:
                mult = (int(word_list[i+1].start_time*100))-last_mult
                last_mult += mult
            else:
                mult = 2

            if word_list[i].text == "p":
                word_text = "<sil>"
            else:
                word_text = word_list[i].text
            
        else:
            word = word_list[i]
            word_text = word.text
            if i != len(word_list)-1:
                mult = (int(word_list[i+1].end_time*100)-1)-last_mult
                last_mult += mult
            else:
                mult = 1
                if word_list[i].text == "<sil>" or word_list[i].text == "p":
                    word_text = lw.text
                else:
                    word_text = word_list[i].text
            lw = word
            
        #print phone
        #print mult

        wordlist += [word_text]*mult
        
    if word_list == []:
        print wordlist
    elif int(wav_len*100) > len(wordlist):
        len_tail = int(wav_len*100) - len(wordlist)
        wordlist += [wordlist[-1]]*len_tail
        
    return wordlist

def get_word_cols(wordlist, cwordlist, wav_len, speaker, ep):

    wav_len = wav_len+1
    
    word_list = get_col(wordlist, wav_len)
    cword_list = get_col(cwordlist, wav_len)
    gold_word_list = get_col(wordlist, wav_len, gold=True)
    gold_cword_list = get_col(cwordlist, wav_len, gold=True)
    sp_list = [int(speaker)]*int(wav_len*100)
    ep_list = [float(ep)]*int(wav_len*100)
    time_in_sec = [(i+1)/100.0 for i in range(int(wav_len*100))]

    return time_in_sec, sp_list, ep_list, gold_word_list, gold_cword_list, word_list, cword_list

def get_words(tgpath):

    word_list = []
    cword_list = []

    textgrid = open_textgrid(tgpath)
    try:
        words = textgrid.get_tier_by_name('words')
    except:
        return [],[]
    #like the phone, dur, zsc generation
    last_word = ""
    cword = "<s>;<s>"
    for word in words:
        w_start = float(word.start_time)
        w_end = float(word.end_time)
        ###################################################
        if word.text=="p":
            if word.start_time == 0.0 or word == words[-1]:
                cword_list.append(tgt.Interval(w_start,\
                                               w_end,\
                                               "<sil>"))
            else:
                cword_list.append(word)
        else:
            cword_list.append(word)
        ###################################################            
        if word.text != last_word and word.text != '<sil>':
            if word.start_time == 0.0 and word.text=="p":
                word_list.append(tgt.Interval(w_start,\
                                               w_end,\
                                               cword))
            else:
                cword += ";"+word.text
                word_list.append(tgt.Interval(w_start,\
                                              w_end, cword))
            if word == words[-1]:
                if word.text == "p":
                    cword = cword[:-2]
                    cword += ";"+"<eot>"
                    word_list.append(tgt.Interval(w_start,\
                                               w_end,\
                                               cword))
                else:
                    cword += ";"+"<eot>"
                    word_list.append(tgt.Interval(w_end,\
                                                   w_end+1501,\
                                                   cword))
        elif word.text == '<sil>':
            if word == words[-1]:
                cword += ";"+"<eot>"
                word_list.append(tgt.Interval(w_start,\
                                               w_end,\
                                               cword))
            else:
                last_word = "<sil>"

    #word_list, cword_list = get_word_cols(word_list, cword_list, wav_len, speaker, ep)
                
    return word_list, cword_list


#read a csv as df and returns a dict with {speaker:episodes}
def get_eps_from_pathslist(eppath_list):
    epdict = {}
    # load dataset as panda-Dataframe
    for eps_path in eppath_list:
        speaker = eps_path.split('_')[0][1]
        ep = int(eps_path.split('_')[1].split('.')[0])
        if not int(speaker) in epdict:
            epdict[int(speaker)] = []
        epdict[int(speaker)].append(int(ep))
    return epdict


#function to read corpus as txt seperated by \n
#args: a filename, the train csv file for prosodic features
#                  to get the same episodes for training,
#      n for n in ngram to add start tags,
#      excluded speaker for training
def read_corpus(fname, eps, n, mtp, exclude=8):
    #cfile = codecs.open(fname,"r","utf8")
    cfile = open(fname)
    #the training corpus
    incorpus = ""
    #the testing corpus for heldout speaker
    excorpus = ""
    print "reading corpus excluding speaker",exclude
    for line in cfile:
        #print "rawline", line
        line = line.strip("\n")
        if line == "":
            continue
        #final MTP is the same as the end of line   
        if line.endswith("MTP"):       
            line = line[:-3]
        if mtp==False:
            line = line.replace("MTP","")
        #clean string of pauses and tags
        line = line.replace(":","").replace("?","").replace("!","").\
                replace(".","").replace(",","").replace("{","").replace("}","").lower()
        line = line.split()
        line = [w for w in line if not "<" in w and not ">" in w and not "cough" in w]
        #exclude a speaker
        speaker = exclude
        #print line[1],line[0]
        #print line
        if int(line[1]) in eps[int(line[0])]:   
            lm_utt_line = " ".join(line[2:]) +"\n" #TODO can add eot here
            if int(line[0]) != speaker:
                incorpus += lm_utt_line
            else:
                excorpus += lm_utt_line
    incorpus = incorpus.encode("utf8")
    excorpus = excorpus.encode("utf8")
    cfile.close()
    #print incorpus
    return incorpus, excorpus

def get_ngrams(corpus, order=3):
    unigrams = set()
    bigrams = set()
    trigrams = set()
    sents = corpus.split("\n")
    tokens = [sent.split() for sent in sents]
    for sent in tokens:
        for i in range(order-1,len(sent)):
            if '="' in sent[i]:
                sent[i] = sent[i][sent[i].find('="')+2 : sent[i].find('">')]
            trigram = sent[i-order+1:i+1]
            for x in trigram:
                unigrams.add(x)
            bigrams.add(tuple(trigram[:2]))
            bigrams.add(tuple(trigram[1:]))
            trigrams.add(tuple(trigram))
    return {"unigram":unigrams, "bigram":bigrams, "trigram":trigrams}

def write_speaker_lm(fname, prob_dict):
    cfile = codecs.open(fname,"w","utf8")
    cfile.write("ngram\t lwbi\t surprisal\n")
    #for gram in prob_dict:
    for ngram in prob_dict:
        cfile.write(str(ngram))
        for val in prob_dict[ngram]:
            cfile.write("\t"+str(val))
        cfile.write("\n")    
    cfile.close()
    return

#get probabilities as list for
#<eot> | w1, w2
#with w2 = lastword (lw) or not (nlw)
#args: dict with two lists, language model, a ngram
#returns the updated dictionary
def get_eot_prob(lm, ngram):
    #eot_prob = {"lw":[],"nlw":[]}
    eot_ngram = list(ngram)
    eot_ngram[2] = '</s>'#"<eot>"
    #print eot_ngram
    lwbi = lm.logprob_weighted_by_inverse_unigram_logprob(eot_ngram)
    prob = lm.ngram_prob(eot_ngram,3)
    return lwbi,prob

def get_mean_and_sd(eot_prob, index=0):
    stats_dict = {"lw":[],"nlw":[]}
    for prob_list in eot_prob:
        problist = [x[index] for x in eot_prob[prob_list]]
        eot_mean = numpy.mean(problist)
        eot_sd = numpy.std(problist)
        stats_dict[prob_list].append(eot_mean)
        stats_dict[prob_list].append(eot_sd)
    return stats_dict

#load dataframe for training data -> eps that are used in training with prosodic features
#read corpora from textfile for training corpus and heldout corpus
#built lm with corpus
#built 3-grams for heldout corpus and get their probabilities applying the lm
def build_lm(fname, eps_dict, exclude, n, mtp):  
    print "training corpus from", fname
    corpora = read_corpus(fname, eps_dict, n, mtp, exclude)
    incorpus = corpora[0]
    #incorpus.encode("utf8")
    incorpus = sorted(corpora[0].split('\n'))
    #shuffle(incorpus) #you can shuffle
    heldout_split = int(len(incorpus)*0.9)
    corpus = "\n".join(incorpus[:heldout_split])
    hcorpus = "\n".join(incorpus[heldout_split:])
    excorpus = corpora[1]
    print "excluded corpus"
    #print excorpus
    #file = open("angelika_lm{}.text".format(exclude),"w")
    #file.write(corpus+"\n\n HELDOUT \n" + hcorpus)
    #file.close()
    #excorpus.encode("utf8")
    #Train a KN smoothed LM which is:
    #3-grams
    #0.7 discounted for unknown words
    #Allows partial words in the model and has special calculation for them
    #has a trainining corpus, and heldout/second copora for estimating prevalence of unknown words
    lm = KneserNeySmoothingModel(order=3,
                                 discount=0.7,
                                 partial_words=True,
                                 train_corpus=corpus,
                                 heldout_corpus=hcorpus,
                                 second_corpus=None)
    return lm

def apply_lm(lm, gold_cwords, gold_words, words, cwords, nspeaker=8, n=2, mtp=False):
    
    ldur_list = [] #duration which the current label applies to
    label_list = [] #list of labels
    #lm feature lists
    wml_list = [] 
    wml_trigram_list = []
    entropy_list = []
   
    counter = 0 
    final_current_words = words[-1] #the target final words
    #initialize variables
    lwbi_wml = None #start values for lm
    lwbi_wml_ngram = None
    lwbi_h = None
    label_dur = 0
    #initialize word prefix to do LM computations on
    last_label = -1
    last_prefix = ""
    for i in range(len(cwords)):
        gold_cword = gold_cwords[i].encode("utf8")
        gold_word = gold_words[i].strip().encode("utf8")
        gold_word = gold_word.split(";")
        prefix =  words[i].strip().encode("utf8")
        #print "prefix", prefix
        
        if words[i] == final_current_words:
            label = 2
        else:
            if gold_cword =='<sil>' or gold_cword == 'p'\
                    or "<cough" in gold_cword:
                label = 0
                #if last label was 1, correct back
                if len(label_list)>0 and label_list[-1] == 1:
                    label_list[-1] = 0
                    ldur_list[-1] = 0.01
                    label_dur = 0.01
            else:
                label = 1
        if label != last_label:
            label_dur = 0.01
        else:
            label_dur+=0.01
        
        if prefix == "<null>": #default values
            lwbi_wml = -3.0
            lwbi_wml_ngram = -3.0
            lwbi_h = 0.0
        elif prefix != last_prefix:
            #only do computation if there's a new word
            eot_all = ['<s>'] * (lm.order-1) + prefix.lower().split(";")[1:] + ['</s>']#+ ["<eot>"]
            eot_all_string = " ".join(eot_all[lm.order-1 : -1]) 
            #print eot_all
            eot_ngram = eot_all[-lm.order:] #the last n
              
            lwbi_wml = lm.logprob_weighted_by_inverse_unigram_logprob(eot_all)
            lwbi_wml_ngram = lm.logprob_weighted_by_inverse_unigram_logprob(eot_ngram)
            lwbi_h = lm.entropy(eot_all_string,lm.order)
        
        label_list.append(label)
        ldur_list.append(label_dur)
        wml_list.append(lwbi_wml)
        wml_trigram_list.append(lwbi_wml_ngram)
        entropy_list.append(lwbi_h)
          
        last_label = label
        last_prefix = prefix
        counter +=1
    return ldur_list, label_list, wml_list, wml_trigram_list, entropy_list


#if __name__ == '__main__':
#    fname = "./utterance_list_julians_lm.txt"
#    csv = "/media/angelika/BACA142ACA13E185/Studium/Masterarbeit/Programmierung/take_the_turn/Code/rms_vad/vad_labeling/data_sets/vad_train_1.csv"
#    #save lm per speaker at:
#    folder = "lm_per_speaker/"
#    n = 2 #start tags <s>
#    model = built_lm(fname, csv, folder, n)

#lwbi stats:
#           mean                     stdev
#{'lw': [-0.71709907107762294, 0.39804025438386215],
#'nlw': [-1.0194009133123081, 0.54423473641278519]}
#prob stats:
#{'lw': [0.21820409406138516, 0.1987126544908342],
#'nlw': [0.14055929151673202, 0.16490988891616482]}
#
