import tensorflow as tf
import random
import pandas as pd
import numpy as np
import math
import json
from tqdm import tqdm, trange
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Flatten, TimeDistributed

###########################################################
# FIXME: Hack required to enable GPU operations by TF RNN
###########################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

##################################
# Constants
##################################
RANDOM_SEED = 7
EMBED_SZ = 64

RNN_BATCH_SZ = 5
RNN_WINDOW_SZ = (
    20  # From Charniak p. 84 ('realistic') -- note that 5x20=100 losses/batch
)
RNN_STATE_SZ = 128
RNN_EPOCHS = 10


def set_seed():
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def txt_tensor(token_list, vocab_dict):
    # Create list of token integer ids; intially all 0 (*UNK*)
    token_ids = np.zeros(len(token_list), dtype=np.int32)
    for i in range(0, len(token_list)):
        token = token_list[i]
        if token in vocab_dict:
            token_ids[i] = vocab_dict[token][0]

    return token_ids


def ngram_tensor(n, token_ids):
    token_ids = token_ids.reshape((len(token_ids), 1))
    concat_list = []
    for i in range(0, n - 1):
        concat_list += [token_ids[i : -n + 1 + i, :]]
    concat_list += [token_ids[n - 1 :, :]]
    return np.concatenate(concat_list, axis=1)


def gen_ngrams(n, file, vocab):
    # Read text file; convert to lower case, split into list (token sequence)
    text = open(file).read().lower().split()
    textSz = len(text)
    raw_token_ids = txt_tensor(text, vocab)
    ngrams = ngram_tensor(n, raw_token_ids)

    return (text, textSz, ngrams)


################################################################
# Tensors for RNN input / answers
################################################################
def ids_2_tokens(id_tensor, wordId_dict):
    # Create a string tensor replacing word ids in input tensor by the tokens.
    # Slow -- not intended for large inputs (more for debugging/visualizing small
    # excerpts)
    tensorShape = id_tensor.shape
    str_nested_list = []
    for batch in range(0, tensorShape[0]):
        for row in range(0, tensorShape[1]):
            row_list = []
            for col in range(0, tensorShape[2]):
                # pcheck("(token id,token)", (id_tensor[row][col], wordId_dict[ id_tensor[row][col]] ))
                row_list.append(wordId_dict[id_tensor[batch][row][col]])
            str_nested_list.append(row_list)
    return str(str_nested_list).replace("],", "],\n")


def gen_rnn_tensor_input(windowSz, batchSz, file, vocab, wordId_dict):
    # Read file; create inputs for RNN
    text = open(file).read().lower().split()
    textSz = len(text)
    raw_token_ids = txt_tensor(text, vocab)

    # Compute tensor with windowSz + 1 elements per row
    # (all input words in window + final predicted word)
    # Make sure we omit words at end if too short to construct another window
    S = math.floor((textSz - 1) / (batchSz * windowSz))
    rows_fitting = S * batchSz * windowSz

    # DEBUG: Needed to remember the stack operation, checked sizes for quite awhile
    rnn_inpt = raw_token_ids[:rows_fitting].reshape([batchSz, S * windowSz])
    rnn_inpt = np.stack(np.hsplit(rnn_inpt, S), axis=0)

    # DEBUG: Labels need to be a 1D list of labels in the correct order.
    rnn_answr = raw_token_ids[1 : rows_fitting + 1].reshape([batchSz, S * windowSz])
    rnn_answr = np.stack(np.hsplit(rnn_answr, S), axis=0)

    return (text, textSz, rnn_inpt, rnn_answr)


if __name__ == "__main__":

    set_seed()

    # Dataset
    vocab = json.load(open('data/vocab_5.json'))
    vocabSz = len(vocab)

    wordId_dict = {}
    for entry in vocab.items():
        wordId_dict[entry[1][0]] = entry[0]

    (text, textSz, rnn_inpt, rnn_answr) = gen_rnn_tensor_input(
        RNN_WINDOW_SZ, RNN_BATCH_SZ, 'data/pony_train.txt', vocab, wordId_dict
    )
    (dev_text, devSz, dev_rnn_inpt, dev_rnn_answr) = gen_rnn_tensor_input(
        RNN_WINDOW_SZ, RNN_BATCH_SZ, 'data/pony_dev.txt', vocab, wordId_dict
    )

    # RNN Architecture for word embeddings
    model = Sequential([
        Embedding(input_dim=vocabSz, output_dim=vocabSz,
                  mask_zero=True, trainable=False, input_length=RNN_WINDOW_SZ,
                  embeddings_initializer=tf.keras.initializers.random_normal()),
        LSTM(units=RNN_STATE_SZ, return_sequences=True),
        Dense(units=vocabSz)
    ], name="RNN cell + linear classification layer")

    model.summary()

    # Define the optimizer and loss
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    # Metrics
    training_loss_tracker = tf.keras.metrics.Mean(name="train_perplexity")
    dev_loss_tracker = tf.keras.metrics.Mean(name="dev_perplexity")

    # Train the model
    train_stats = trange(0, desc='train_perplexity')
    dev_stats = trange(0, desc='dev_perplexity')
    batch_count_train = rnn_inpt.shape[0]
    batch_count_dev = dev_rnn_inpt.shape[0]

    for epoch in tqdm(range(0, RNN_EPOCHS)):

        # Training loss/perplexity
        for batch in range(0, batch_count_train):
            labelMatrix = rnn_answr[batch, :, :]
            labelVector = np.reshape(labelMatrix, [RNN_BATCH_SZ * RNN_WINDOW_SZ])

            with tf.GradientTape() as tape:
                x = rnn_inpt[batch, :, :]
                y_pred = model(x)
                total_loss = loss(labelVector, y_pred)

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            training_loss_tracker.update_state(total_loss)
            train_perplexity = math.exp(training_loss_tracker.result())
            train_stats.set_description('train_perplexity=%g' % train_perplexity)

        # Development loss/perplexity
        for batch in range(0, batch_count_dev):
            labelMatrix = dev_rnn_answr[batch, :, :]
            labelVector = np.reshape(labelMatrix, [RNN_BATCH_SZ * RNN_WINDOW_SZ])

            # Do not train in the dev set
            x = dev_rnn_inpt[batch, :, :]
            y_pred = model(x)
            total_loss = loss(labelVector, y_pred)

            dev_loss_tracker.update_state(total_loss)
            dev_perplexity = math.exp(dev_loss_tracker.result())
            dev_stats.set_description('dev_perplexity=%g' % dev_perplexity)

