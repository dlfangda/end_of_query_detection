acoustic_features = 'label ~ pcm_RMSenergy_sma + rms_minus_one + rms_minus_two + rms_minus_three + rms_minus_four +\
pcm_LOGenergy_sma + pcm_loudness_sma + pcm_intensity_sma  +\
zscore + duration + intensity_mean + intensity_slope +\
F0final_sma + F0raw_sma + F0_mean + F0_slope'

#acoustic_features = 'label ~ time_in_sec + \
#pcm_RMSenergy_sma + rms_minus_one + rms_minus_two + rms_minus_three + rms_minus_four +\
#pcm_LOGenergy_sma + pcm_loudness_sma + pcm_intensity_sma  +\
#intensity_mean + intensity_slope'

 ######################################################
lm_features = 'label ~ wml + wml_trigram + entropy'
