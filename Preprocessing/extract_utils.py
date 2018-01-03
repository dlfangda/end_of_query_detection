#!python

import numpy as np
import pandas as pd
import os
import tgt
import wave
import codecs
import cPickle as pickle

#args: rootdir -> root directory, ending -> file ending
#return: list of pathes in rootdir 
def lsdir(rootdir, ending):
    pathlist = []
    for root,  dirs,  files in os.walk(rootdir,  topdown=False):
        for filename in files:
            if ending in filename:
                pathlist.append(filename)
    return pathlist

#arg: tgfile -> textgrid filepath
#return: if IOError None eles textgrid object
def open_textgrid(tgfile):
    #print "open file: "+tgfile
    try:
        textgrid = tgt.read_textgrid(tgfile, encoding = 'utf-8')
        return textgrid
    except:
        try:
            textgrid = tgt.read_textgrid(tgfile, encoding = 'utf-16')
            return textgrid        
        except IOError:
            print "Textgrid not found"
            pass
            return

def open_pkl_ep(speaker,ep,ep_path):
    fname = ep_path+'r'+str(int(speaker))+'_'+str(int(ep))+'.pkl'
    with open(fname,'rb') as fp:
        ep_df = pickle.load(fp)
    #print fname
    #ep_df = pickle.load(fname)
    return ep_df

def pkl_dataframe(df, out_path):
    with open(out_path,'wb') as nfid:
        pickle.dump(df, nfid, protocol=pickle.HIGHEST_PROTOCOL)
    return

def mean(liste):
    m = sum(liste)/len(liste)
    return m

def sd(liste):
    prevariance = []
    mean_len = mean(liste)

    for plen in liste:
        prevariance.append(float((plen - mean_len)**2))

    if len(liste) > 1:
        stdv = (1.0/(len(liste)-1))*sum(prevariance)
    else:
        stdv = 0.0
    
    return stdv

def get_phone_set(tgpath, vowels):

    ep_id = tgpath.split("/")[-1].split(".")[0]
    
    tg = open_textgrid(tgpath)
    pho = tg.get_tier_by_name("phones")

    phones = []
    for p in pho:
        if p.text in vowels:
            phones.append(p)
            #here you also get the duration of the vowel
            #take the end_time-10 and just append the current values until the next end_time-10
        
    return phones

def get_phones(tgpath, phones):

    ep_id = tgpath.split("/")[-1].split(".")[0]
    
    tg = open_textgrid(tgpath)
    pho = tg.get_tier_by_name("phones")

    for p in pho:
        if p.text not in phones:
            phones[p.text] = []
        phones[p.text].append(ep_id)

    return phones

def get_duration(tgpath, phones):    

    ep_id = tgpath.split("/")[-1].split(".")[0]
    
    tg = open_textgrid(tgpath)
    pho = tg.get_tier_by_name("phones")
    
    for p in pho:
        if p.text not in phones:
            phones[p.text] = []
        dur = p.end_time - p.start_time
        phones[p.text].append(dur)
        
    return phones


def get_col(word_list, ep_len, gold=False):
    """Loops over tgt intervals with words in them.
    If gold is true, then return the simulated ASR
    results where in the last frame of the word event
    the value of the word becomes available.
    
    """
    wordlist = [] #list of latest words
    wordprefixlist = [] #list of all the words so far
    last_word = "<null>"
    last_prefix = "<null>"
    last_mult = 0
    for i in range(len(word_list)):
        if gold:
            if i != len(word_list)-1:
                mult = (int(word_list[i+1].start_time*100))-last_mult
                last_mult += mult
            else:
                mult = 2

            if word_list[i].text == "p":
                word_text = "<sil>"
            else:
                word_text = word_list[i].text
            last_prefix+=";"+word_text
            wordprefixlist += [last_prefix]*mult
            wordlist += [word_text]*mult
        else:
            word_text = word_list[i].text.replace(":","").\
                    replace("!","").replace("?","").replace(".","")
            if i != len(word_list)-1:
                mult = (int(word_list[i+1].end_time*100)-1)-last_mult
                last_mult += mult
            else:
                mult = 1
            if word_text == "<sil>" or word_text == "p" or "<" in word_text or ">" in word_text:
                #use last value
                word_text = last_word
            else:
                last_prefix+=";"+word_text
            last_word = word_text
            wordprefixlist += [last_prefix]*mult
            wordlist += [word_text]*mult
        
    if word_list == []:
        print wordlist
    elif ep_len > len(wordlist):
        len_tail = ep_len - len(wordlist)
        wordlist += [wordlist[-1]]*len_tail
        wordprefixlist +=[wordprefixlist[-1]]*len_tail
        
    return wordlist, wordprefixlist

def get_word_cols(wordlist, cwordlist, ep_len, speaker, ep):

    cword_list, words_list = get_col(cwordlist, ep_len) #simulated ASR word
    gold_cword_list, gold_word_list = get_col(cwordlist, ep_len, gold=True) #ground truth from textgrid words
    sp_list = [int(speaker)]*int(ep_len)
    ep_list = [float(ep)]*int(ep_len)
    time_in_sec = [(i+1)/100.0 for i in range(int(ep_len))]

    return time_in_sec, sp_list, ep_list, gold_word_list, gold_cword_list, words_list, cword_list

def get_words(tgpath):
    word_list = [] #list of the gold word tgt intervals according to the textgrid (history)
    cwords_list = [] #list of the right frontier word (latest) as tgt interval

    textgrid = open_textgrid(tgpath)
    try:
        words = textgrid.get_tier_by_name('words')
    except:
        return [],[]
    #like the phone, dur, zsc generation
    last_word = ""
    all_words = "<s>;<s>"
    raw_cword_list = list(words)
    cword_list = []    
    for word in raw_cword_list:
        #print word
        if word.text == "p":
            word.text = "<sil>"
        cword_list.append(tgt.Interval(word.start_time, word.end_time, word.text))
    for i in range(len(cword_list)):
        w_start = cword_list[i].start_time
        w_end = cword_list[i].end_time
        w_text = cword_list[i].text
        #print w_text
        if i == 0:
            if cword_list[i].text == "<sil>" or w_text == "p":
                word_list.append(tgt.Interval(w_start, w_end, all_words))
                cwords_list.append(tgt.Interval(w_start, w_end, "<sil>"))
            else:
                all_words += ";"+w_text
                word_list.append(tgt.Interval(0.0, w_start, "<s>;<s>"))                
                word_list.append(tgt.Interval(w_start, w_end, all_words))
                
                cwords_list.append(tgt.Interval(0.0, w_start, "<sil>"))
                cwords_list.append(tgt.Interval(w_start, w_end, w_text))
                
        elif i == len(cword_list)-1:
            if cword_list[i].text == "<sil>" or cword_list[i].text =="p":
                all_words += ";"+"</s>"
                word_list.append(tgt.Interval(w_start, w_end, all_words))
                
                cwords_list.append(tgt.Interval(w_start, w_end, "<sil>"))                
            else:
                all_words += ";"+w_text
                word_list.append(tgt.Interval(w_start, w_end, all_words))
                all_words += ";"+"</s>"
                word_list.append(tgt.Interval(w_end, w_end+15.01, all_words))
                
                cwords_list.append(tgt.Interval(w_start, w_end, w_text))
                cwords_list.append(tgt.Interval(w_end, w_end+15.01, "<sil>"))
        else:
            all_words += ";"+w_text
            word_list.append(tgt.Interval(w_start, w_end, all_words))
            cwords_list.append(tgt.Interval(w_start, w_end, w_text))
    return word_list, cwords_list

def get_words_old(tgpath):

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
            #else:
            #    cword_list.append(word)
        else:
            cword_list.append(word)
        ###################################################            
        if word.text != last_word and word.text != '<sil>':
            if word.start_time == 0.0 and word.text=="p":
                word_list.append(tgt.Interval(w_start,\
                                               w_end,\
                                               cword))
            else:
                if word.text != "p":
                    cword += ";"+word.text
                    word_list.append(tgt.Interval(w_start,\
                                                  w_end, cword))
            if word == words[-1]:
                if word.text == "p":
                    cword = cword[:-2]
                    cword += ";"+"</s>"
                    word_list.append(tgt.Interval(w_start,\
                                               w_end,\
                                               cword))
                else:
                    cword += ";"+"</s>"
                    word_list.append(tgt.Interval(w_end,\
                                                   w_end+1501,\
                                                   cword))
        elif word.text == '<sil>':
            if word == words[-1]:
                cword += ";"+"</s>"
                word_list.append(tgt.Interval(w_start,\
                                               w_end,\
                                               cword))
            else:
                last_word = "<sil>"
            
    return word_list, cword_list


def get_durations_for_ep(tgpath):
    
    dur_list = []
    phone_list = []
    
    vowels = [u'Y', u'y:', u'u:', u'aI', u'aU', u'o:',\
              u'9', u'i:', u'E', u'y:6', u'I', u'E:', u'O', \
              u'U',  u'e:', u'a:', u'a',  u'e', u'i',\
              u'o', u'u', u'OY', u'6', u'@']
    #including nasals and fricatives
    vowels_plus = vowels + [u'n', u'm', u'f', u's', u'S']
    vowels = vowels_plus

    #schwa = [u'6', u'@']
    #nasal = [u'm', u'l', u'n']
    #cons = [u'?', u'C', u'N', u'S', u'b', u'd', u'g', u'f',\
    #        u'h', u'k', u'j', u'p', u's', u'r', u't', u'v', u'x', u'z']
    #other = [u'<unclear/>', u'<nonverbal/>',\
    #         u'<laughter/>', u'<coughing/>', u'<p:>']

    ep_id = tgpath.split("/")[-1].split(".")[0]
    ep = ep_id.split("_")[1]
    fnum = ep_id.split("_")[0][1:]
    #print fnum
    #print ep_id
    
    tg = open_textgrid(tgpath)
    pho = tg.get_tier_by_name("phones")
    words = tg.get_tier_by_name("words")
    words = [tgt.Interval(w.start_time,w.end_time,w.text)\
             for w in words if w.text != '<sil>' and w.text != 'p']

    ep_len = int((words[-1].end_time+15.0)*100)

    vowel_set = [p for p in pho if p.text in vowels]
    vowel_set = [tgt.Interval(0.0,vowel_set[0].start_time,"nan")]+vowel_set
    
    last_mult = 0
    
    for i in range(len(vowel_set)):
        if i != len(vowel_set)-1:
            mult = (int(vowel_set[i+1].end_time*100)-1)-last_mult
            last_mult += mult
        else:
            mult = 2
        phone = vowel_set[i]
        #print phone
        #print mult

        if i == 0:            
            pho_dur = 0.0            
        else:
            pho_dur = phone.end_time - phone.start_time
            
        phone_list += [phone.text]*mult
        dur_list += [pho_dur]*mult
    #print len(ep_df), len(dur_list)
    #print len(ep_df)-len(dur_list)
    
    if ep_len > len(dur_list):
        len_tail = ep_len - len(dur_list)
        dur_list += [dur_list[-1]]*len_tail
        phone_list += [phone_list[-1]]*len_tail

    return dur_list, phone_list, ep_len


def get_pho_dicts(prepath, tgfiles):

    lastfn = "r1"
    #phone_dict = {}
    dur_dict = {}
    dur_dict_all = {}
    final_all = {}
    liste = ["<p:>","OY","aI","aU","<coughing/>","<laughter/>","<unclear/>","<nonverbal/>"]
    
    for i in range(len(tgfiles)):
        filenumber = tgfiles[i].split(".")[0][:2]
        if filenumber != lastfn:
            print filenumber
            lastfn = filenumber
            dur_dict[filenumber] = {}            
        #phone_dict = get_phones(prepath+filenumber+"/"+tgfiles[i], phone_dict)
        dur_dict[filenumber] = get_duration(prepath+filenumber+"/"+tgfiles[i], dur_dict[filenumber])
        dur_dict_all = get_duration(prepath+filenumber+"/"+tgfiles[i], dur_dict_all)
        
    return dur_dict

def get_zscore(val, val_mean, val_sd):
    if val_sd != 0.0:
        zscore = (val-val_mean)/val_sd
    else:
        zscore = 0.0
    return zscore

def get_durations_for_speaker(tg_path, speaker, eps_dict):
    out_list = ["<p:>","OY","aI","aU","<coughing/>","<laughter/>","<unclear/>","<nonverbal/>"]
    zsc_dict = {speaker:{}}

    for ep in eps_dict:#[speaker]:
        #print ep
        path = tg_path + "/r"+str(speaker)+"/r"+str(speaker)+"_"+str(ep)+".TextGrid"
        zsc_dict[speaker] = get_duration(path, zsc_dict[speaker])
                
    return zsc_dict

def get_zsc_per_ep(ep_df, dur_dict):

    ep_dur = ep_df.duration.tolist()
    ep_phones = ep_df.phones.tolist()
    zsc_list = []
    ldur = -1
    for i in range(len(ep_dur)):
        
        if ep_dur[i] != ldur and ep_dur[i] != 0.0:
            
            val_mean = np.mean(dur_dict[ep_phones[i]])
            val_sd = np.std(dur_dict[ep_phones[i]])
            zsc = get_zscore(ep_dur[i], val_mean, val_sd)

        elif ep_dur[i] == 0.0:
            zsc = -10            

        zsc_list.append(zsc)
        
    return zsc_list

#get vowel values when it is uttered
def get_dur_zsc(tgpath, wav_len, dur_dict):
    
    dur_list = []
    zsc_list = []
    phone_list = []
    
    vowels = [u'Y', u'y:', u'u:', u'aI', u'aU', u'o:',\
              u'9', u'i:', u'E', u'y:6', u'I', u'E:', u'O', \
              u'U',  u'e:', u'a:', u'a',  u'e', u'i',\
              u'o', u'u', u'OY', u'6', u'@']
    vowels2 = [u'Y', u'y:', u'u:', u'aI', u'aU', u'o:',\
              u'9', u'i:', u'E', u'y:6', u'I', u'E:', u'O', \
              u'U',  u'e:', u'a:', u'a',  u'e', u'i',\
              u'o', u'u', u'OY', u'6', u'@', u'n', u'm', u'f', u's', u'S']
    vowels = vowels2
    #schwa = [u'6', u'@']
    #nasal = [u'm', u'l', u'n']
    #cons = [u'?', u'C', u'N', u'S', u'b', u'd', u'g', u'f',\
    #        u'h', u'k', u'j', u'p', u's', u'r', u't', u'v', u'x', u'z']
    #other = [u'<unclear/>', u'<nonverbal/>',\
    #         u'<laughter/>', u'<coughing/>', u'<p:>']

    ep_id = tgpath.split("/")[-1].split(".")[0]
    ep = ep_id.split("_")[1]
    fnum = ep_id.split("_")[0][1:]
    #print fnum
    #print ep_id
    
    tg = open_textgrid(tgpath)
    pho = tg.get_tier_by_name("phones")
    words = tg.get_tier_by_name("words")
    
    vowel_set = get_phone_set(tgpath, vowels)
    vowel_set = [tgt.Interval(0.0,vowel_set[0].start_time,"nan")]+vowel_set
    
    last_mult = 0
    
    for i in range(len(vowel_set)):
        if i != len(vowel_set)-1:
            mult = (int(vowel_set[i+1].end_time*100)-1)-last_mult
            last_mult += mult
        else:
            mult = 2
        phone = vowel_set[i]
        #print phone
        #print mult

        if i == 0:            
            pho_dur = 0.0
            zscore = -10
        else:
            pho_dur = phone.end_time - phone.start_time
            val_mean = np.mean(dur_dict[phone.text])
            val_sd = np.std(dur_dict[phone.text])
            zscore = get_zscore(pho_dur, val_mean, val_sd)  
        
        phone_list += [phone.text]*mult
        dur_list += [pho_dur]*mult
        zsc_list += [zscore]*mult
    
    #print len(ep_df), len(dur_list)
    #print len(ep_df)-len(dur_list)
    
    if wav_len > len(dur_list):
        len_tail = wav_len - len(dur_list)
        dur_list += [dur_list[-1]]*len_tail
        zsc_list += [zsc_list[-1]]*len_tail
        phone_list += [phone_list[-1]]*len_tail

    return dur_list, zsc_list, phone_list

#Functions to extract the utterances from the TextGrids
#Necessary for Language Model
def replace_markers(string, mark_lengthening="off"):
    out_markers = [")","(","+","!","-"]
    if mark_lengthening != "off":
        out_markers.append(":")
    markers = ["...","..","."]
    for marker in out_markers:
        string = string.replace(marker,"")
    for marker in markers:
        string = string.replace(marker," MTP ")
    return string

######################################################
def get_real_utt(string):
    liste = string.split()
    real_utt = ""
    for word in liste:
        try:
            if "=" in word:
                real_word = word.split('=""')[1]
                real_word = real_word.split('"">')[0]
                real_utt += real_word+" "            
            else:
                if not "<" in word:
                    real_utt += word+" "
        except:
            print unicode(word)
    real_utt = real_utt.strip()
    
    return real_utt

#sequence of p and Instruct
def utt_for_file(tgpath, utttxt):
    #print tgpath
    textgrid = open_textgrid(tgpath)
    #new key in tag_dict and get text from utterance-tier
    real_utts = []
    utts = textgrid.get_tier_by_name('utterances')
    for utt in utts:
        if utt.text!="p" and utt.text!="trp":
            utt_text = replace_markers(utt.text)  
            real_utt = get_real_utt(utt_text)
            if real_utt != "":
                real_utts.append(real_utt.strip())
    line = " MTP ".join(real_utts)+"\n"
    line = line.replace("MTP MTP MTP","MTP")
    line = line.replace("MTP MTP","MTP")
    return line
