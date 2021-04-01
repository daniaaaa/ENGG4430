Requires Python 3.6.x
This is due to all required libraries being compatable with python 3.6.
May also work with Python 3.7 and 3.8 (not tested) but is not compatable with python 3.9 or higher as tensorflow is not compatible with python 3.9

Code is not original and has been repurposed from the following sources:
Deepspeech Library + examples
https://github.com/mozilla/DeepSpeech
https://github.com/mozilla/DeepSpeech-examples
https://deepspeech.readthedocs.io/en/v0.9.3/USING.html

Spam Detection Tutorial
https://towardsdatascience.com/nlp-detecting-spam-messages-with-tensorflow-b12195b8cf0e
https://github.com/MGCodesandStats/tensorflow-nlp

Spam data obtained from online database on Kaggle:
https://www.kaggle.com/team-ai/spam-text-message-classification


Not included but still required are the speech recognition models a .scorer and a .pbmm.
Current version of the project utilizes models from deepspeech library v 0.9.3
Pretrained Models at:
https://deepspeech.readthedocs.io/en/v0.9.3/USING.html
Code to download models:
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

Instructions to run live spam detection:
Live Run Commands
	- Enter project directory
	- On command line enter the following:
		- Python -m venv . (creates virtual environment, requires virtualenv python library)
		- Scripts\activate (activates virtual environment, may need to edit script permissions in order to run, see below)
			○ Activation may require new permissions to be set for windows scripts
			○ https://virtualenv.pypa.io/en/legacy/userguide.html#activate-script
			○ Further details in "Adjusting Permssions to run virtual environment" sectio below
	- Pip install deepspeech
	- cd mic_vad_streaming
		○ Pip install -r requirements.txt
			§ If fail on pyaudio or any other requirements manually install with .whl file at:
				- https://www.lfd.uci.edu/~gohlke/pythonlibs/#pandas
	- Python liveSpamDetect.py -m (path to .pbmm) -s (path to scorer)
		Example command to run code with models inside mic_vad_streaming folder
		○ Python liveSpamDetect.py -m deepspeech-0.9.3-models.pbmm -s deepspeech-0.9.3-models.scorer
	- Ctrl - c exits program
	- Need to speak slowly and clearly to get best speech recognition
	- For each statement recognized the program will output the thought likelyhood of the call being a scam


Instructions to run live voice speech to text:
	- follow the same steps as above for setup but us the following command instead:
	- Python mic_vad_streaming.py -m (path to .pbmm) -s (path to scorer)
		Example command to run code with models inside mic_vad_streaming folder
		○ Python mic_vad_streaming.py -m deepspeech-0.9.3-models.pbmm -s deepspeech-0.9.3-models.scorer
	- Ctrl - c exits program
	- Need to speak slowly and clearly to get best speech recognition


Adjusting Permissions to run virtual environment:
	- Virtual environment
		○ Pip install venv
		○ To activate virtual environment need to ensure script has permission to run
			§ https://stackoverflow.com/questions/48148664/the-term-execution-policy-is-not-recognized/48160516
	- Within commandline enter the following:	
		- Powershell.exe (if not already in powershell)
		- get-executionpolicy -list (lists script run permissions)
		- set-executionpolicy Allsigned (sets script run permissions)
		- cmd.exe (returns to commandline)


Additional required libraries include the following:
import io
import tensorflow as tf
import numpy as np
import pandas as pd
import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import deepspeech
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
