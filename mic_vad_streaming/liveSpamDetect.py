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
from mic_vad_streaming import Audio
from mic_vad_streaming import VADAudio

#combination of main from mic_vad_streaming.py and spam detection
def main(ARGS):
    # load in trained models for spam detection
    # retrieve training data
    # category 0 is genuine and 1 is spam
    # dataset is stored in "files" section of this notebook
    # may need to reupload dataset upon runtime refresh
    dataset = pd.read_csv('spamData.csv')
    dataset

    # parsing data
    # spam = 1, genuine = 0
    sentences = dataset['Message'].tolist()
    labels = dataset['Category'].tolist()

    # Separate out the sentences and labels into training and test sets
    training_size = int(len(sentences) * 0.8)

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    # Make labels into numpy arrays for use with the network later
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    #tokenize data and pad entries such that they are equivalent length
    vocab_size = 600
    embedding_dim = 16
    max_length = 60
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences,maxlen=max_length, padding=padding_type, 
                          truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences,maxlen=max_length, 
                                  padding=padding_type, truncating=trunc_type)

    # review a sequence to ensure tokenization was successful
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    print(decode_review(padded[1]))
    print(training_sentences[1])

    # Train text classification model with embeddings
    # Note the embedding layer is first, 
    # and the output is only 1 node as it is either 0 or 1 (negative or positive)
    modelSpam = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    modelSpam.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    modelSpam.summary()

    # visual modelling
    num_epochs = 30
    history=modelSpam.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

    # First get the weights of the embedding layer
    e = modelSpam.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape) # shape: (vocab_size, embedding_dim)
    
    #############################################################
    # Write out the embedding vectors and metadata
    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for word_num in range(1, vocab_size):
      word = reverse_word_index[word_num]
      embeddings = weights[word_num]
      out_m.write(word + "\n")
      out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()

    ###############################################################
    # Load DeepSpeech model
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, 'output_graph.pb')
        ARGS.scorer = os.path.join(model_dir, ARGS.scorer)

    print('Initializing model...')
    logging.info("ARGS.model: %s", ARGS.model)
    model = deepspeech.Model(ARGS.model)
    if ARGS.scorer:
        logging.info("ARGS.scorer: %s", ARGS.scorer)
        model.enableExternalScorer(ARGS.scorer)

    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=ARGS.vad_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate,
                         file=ARGS.file)
    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()
    callTranscript = ['Start of Call:']

    # Stream from microphone to DeepSpeech using VAD
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner='line')
    stream_context = model.createStream()
    wav_data = bytearray()
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            logging.debug("streaming frame")
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            if ARGS.savewav: wav_data.extend(frame)
        else:
            if spinner: spinner.stop()
            logging.debug("end utterence")
            if ARGS.savewav:
                vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                wav_data = bytearray()
            text = stream_context.finishStream()
            callTranscript.append(text)
            # save stream buffer and send through spam detection
            # Create the sequences
            padding_type='post'
            sample_sequences = tokenizer.texts_to_sequences(callTranscript)
            fakes_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)           

            classes = modelSpam.predict(fakes_padded)

            # The closer the class is to 1, the more likely that the message is spam
            for x in range(len(callTranscript)):
              print("Recognized: %s" % callTranscript[x])
              print("Spam Liklihood: ")
              print(classes[x])
            print('\n')
          
            stream_context = model.createStream()

if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 16000

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-v', '--vad_aggressiveness', type=int, default=3,
                        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")
    parser.add_argument('-w', '--savewav',
                        help="Save .wav files of utterences to given directory")
    parser.add_argument('-f', '--file',
                        help="Read from .wav file instead of microphone")

    parser.add_argument('-m', '--model', required=True,
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-s', '--scorer',
                        help="Path to the external scorer file.")
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")

    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)
