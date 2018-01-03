#!python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def gen_trans_file(trans_dict, fname):
    trans_df = pd.DataFrame()
    #row names
    rnames = sorted(trans_dict.keys())
    del rnames[rnames.index('se')]
    cnames = sorted(trans_dict.keys())
    del cnames[rnames.index('s')]
    trans_df['1-2->'] = rnames

    col_list = []

    for tran in rnames:
        #tran is the column name
        targets = trans_dict[tran]
        list_vals = []
        ind_list = []
        for i in range(len(rnames)):
            list_vals.append(0)
        for targ in targets:
            if targ == 'se': targ = 's'
            ind_list.append(rnames.index(targ))
        for ind in ind_list:
            list_vals[ind] = 1
        col_list.append(list_vals)
        #print tran, list_vals

    for tran in rnames:
        if tran != 's':
            trans_df[tran] = [val[rnames.index(tran)] for val in col_list]
        else:
            trans_df['se'] = [val[rnames.index(tran)] for val in col_list]

    #trans_df = trans_df.unstack(0)
    #print trans_df
    trans_df.to_csv(fname,sep='\t',index=False)
    return

trans_dict = {}
trans_dict['s'] = ['mtp']
trans_dict['trp'] = ['trp','se']
trans_dict['mtp'] = ['speech','mtp']
trans_dict['speech'] = ['speech','mtp','trp']
trans_dict['se'] = ['trp']
folder = '/media/angelika/BACA142ACA13E185/Studium/Masterarbeit/Programmierung/take_the_turn/Code/hmm_decoder/models/'
fname = folder+'trp_auto.csv'

#gen_trans_file(trans_dict, fname)
