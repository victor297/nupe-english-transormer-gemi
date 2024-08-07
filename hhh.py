import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
import streamlit as st

# Load the dataset
data = pd.read_csv('nupe.csv')

# Split the dataset into input (Nupe) and output (English) sentences
input_texts = data['nupe'].values
target_texts = data['eng'].values

# Tokenize the input and output texts
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
input_sequences = pad_sequences(input_sequences, padding='post')

target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
target_sequences = pad_sequences(target_sequences, padding='post')

input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

max_input_len = input_sequences.shape[1]
max_target_len = target_sequences.shape[1]

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_vocab_size, 256)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(target_vocab_size, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare target data for training
target_sequences_input = target_sequences[:, :-1]
target_sequences_output = target_sequences[:, 1:]

# Add a dimension to the target data to match the model's output shape
target_sequences_output = np.expand_dims(target_sequences_output, -1)

# Train the model
model.fit([input_sequences, target_sequences_input], target_sequences_output, batch_size=64, epochs=100, validation_split=0.2)

# Save the model
model.save('nupe_to_english.h5')

# Load the model
model = tf.keras.models.load_model('nupe_to_english.h5')

# Define the inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_tokenizer.word_index['start']

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_tokenizer.index_word[sampled_token_index]
        decoded_sentence += ' ' + sampled_word

        # Exit condition: either hit max length or find stop character.
        if (sampled_word == 'end' or
           len(decoded_sentence) > max_target_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

# Streamlit app
st.title('Nupe to English Translation')
nupe_text = st.text_input('Enter Nupe text:')

if st.button('Translate'):
    input_sequence = input_tokenizer.texts_to_sequences([nupe_text])
    input_sequence = pad_sequences(input_sequence, maxlen=max_input_len, padding='post')
    translation = decode_sequence(input_sequence)
    st.write('Translation:', translation)
