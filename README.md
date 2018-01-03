Code for running Interspeech 2017 submission experiments on end-of-turn detection. Please cite the following paper if you use this code:

```
@inproceedings{MaierEtAl17InterSpeech,
  author       = {Maier, Angelika and Hough, Julian and Schlangen, David},
  booktitle    = {Proceedings of INTERSPEECH 2017},
  location     = {Stockholm, Sweden},
  title        = {{Towards Deep End-of-Turn Prediction for Situated Spoken Dialogue Systems}},
  year         = {2017},
}
```

Note: this has only been tested on Ubuntu 14.

To get the data needed please make sure to clone [https://bitbucket.org/anmaier/eot_detection_data] as a sister directory to this one. This code makes use of the data in that directory.

Firstly, make sure `Python 2.7` is installed and is your default python installation on your machine or envrionment. If you want to run the results plotting notebooks you need `IPython` installed.

Secondly, install Cython, then h5py, by running the below on the command line:

`sudo pip install Cython`

`sudo pip install h5py`

You then need to run the below from the command line from inside this folder:

`sudo pip install -r requirements.txt`

We use the theano back-end to Keras. To ensure this is set correctly, you need to edit your local keras config, normally at `~/.keras/keras.json` such that the back-end line should be `"backend": "theano"`.

To reproduce the experiments in full it is possible to simply run the following three scripts.

If you have a few hours to spare and want to train the LSTM, run the below, however the models are stored so you may want to skip running the following:

`python Training/train_models.py`

To get the final outputs on the folds as described in the paper with the saved model or the one you've just trained, you then need to run the below:

```
python Evaluation/get_probability_distributions_from_distributions.py
python Evaluation/get_final_output.py
```

You will see the final accuracy results at the end in the files `Evaluation/lstm_5_folds.text` and `Evaluation/lstm_5_folds_r[n].text` where `n` is in range {2,7} for each speaker.

The plot of the overall results can be seen by running the IPython notebook `Evaluation/InterSpeech2017_Results_Plot.ipynb`.

To plot an individual episode's labels and features, run the IPython notebook `Evaluation/Plot_labels_and_features_for_an_episode.ipynb`




#EXTRA: To run your own scripts including extraction from raw audio:

To extract audio features, you must install OpenSmile. On Linux this can be done by dowloading:

[http://sourceforge.net/projects/opensmile/files/opensmile-2.0-rc1.tar.gz/download]

Unzip that folder and run the bash scripts according to the README.

If you are doing this on Mac, please make sure to replace the buildWithPortAudio.sh script with the one designed for Mac by Hendrik Buschmeier at:

[https://gist.github.com/hbuschme/6456249]

You can run the below with the audio to your opensmile distribution's bin file and the folder with audio (wav) files as arguments e.g. 

`python Preprocessing/preprocess_data.py -opensmile /Applications/opensmile-2.0-rc1/opensmile/inst/bin/SMILExtract -audio /path/to/my/audio/files`

#FUTURE:
To run live VAD you need to install webrtcvad according to [https://github.com/wiseman/py-webrtcvad/].
