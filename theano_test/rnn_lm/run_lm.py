
# coding: utf-8

import numpy as np
import cPickle
import time
import sys
import os
import operator
import io
import argparse
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from datetime import datetime
from gru import GRUTheano
from optimizer import *

SENTENCE_START_TOKEN = "<bos>"
SENTENCE_END_TOKEN = "<eos>"
UNKNOWN_TOKEN = "<UNK>"

def get_word_freq(sentences):
    word_dict = {}
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = 0
            word_dict[word] += 1
    return word_dict

def load_data(filename="data/select_top_3000_cn", vocabulary_size=5000, min_sent_characters=0):

    word_to_index = []
    index_to_word = []

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading data file...")
    with open(filename, 'r') as f:
        sentences = [s.strip() for s in f if len(s) > min_sent_characters]
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
        sentences = [sent.split(" ") for sent in sentences]
    print("Parsed %d sentences." % (len(sentences)))
    
    if not os.path.isfile(filename + ".vocab.pkl"):
        # Count the word frequencies
        word_freq = get_word_freq(sentences)
        print("Found %d unique words tokens." % len(word_freq.items()))
        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size-2]
        cPickle.dump(vocab, open(filename + ".vocab.pkl", "w"))
    else:
        vocab = cPickle.load(open(filename + ".vocab.pkl", "r"))
    print("Using vocabulary size %d." % len(vocab))
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
    index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(sentences):
        sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in sentences])

    return X_train, y_train, word_to_index, index_to_word

def sample_multinomial(p):
    """
       Sample multinomial distribution with parameters given by p
       Returns an int    
    """
    x = np.random.uniform(0, 1)
    for i,v in enumerate(np.cumsum(p)):
        if x < v: return i
    return len(p) - 1 # shouldn't happen...

def generate_sentence(model, index_to_word, word_to_index, min_length=5):
    # We start the sentence with the start token
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(new_sentence)[-1]
        sampled_word = sample_multinomial(next_word_probs)
        #samples = np.random.multinomial(1, next_word_probs)
        #sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        if len(new_sentence) > 100 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
            return None
    if len(new_sentence) < min_length:
        return None
    return new_sentence

def generate_sentences(model, n, index_to_word, word_to_index):
    for i in range(n):
        sent = None
        while not sent:
            sent = generate_sentence(model, index_to_word, word_to_index)
        sentence_str = [index_to_word[x] for x in sent[1:-1]]
        print(" ".join(sentence_str))
        sys.stdout.flush()

def train(args):
    # Load data
    x_train, y_train, word_to_index, index_to_word = load_data(args.dataset, args.vocab_size)
    print "building model..."
    model = GRUTheano(len(index_to_word), hidden_dim=args.hidden_dim, bptt_truncate=-1)
    if args.reload:
        model.load_params(args.model)
    print "building f_grad and f_update..."
    lr = T.scalar(name='lr')
    f_grad, f_update = sgd(lr, model.params, model.grads, [model.x, model.y], model.cost)
    print "building model done."
    # Print SGD step time
    t1 = time.time()
    cost = f_grad(x_train[10], y_train[10])
    f_update(args.learning_rate)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
    sys.stdout.flush()
    
    for epoch in range(20):
        num_examples_seen = 0
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            cost = f_grad(x_train[i], y_train[i])
            if np.isnan(cost):
                print 'NaN detected: ', i 
                continue
            if np.isinf(cost):
                print 'inf detected: ', i 
                continue 
            f_update(args.learning_rate)
            num_examples_seen += 1
            # Optionally do callback
            if (args.print_every and num_examples_seen % args.print_every == 0):
                print ("epoch %d, num_examples_seen %d" % (epoch,  num_examples_seen))
                dt = datetime.now().isoformat()
                loss = model.calculate_loss(x_train[:10000], y_train[:10000])
                print("\n%s [loss]: %f" % (dt, loss))
                model.save_params(args.model)
                print ("model saved in %s done." % args.model)
                print("--------------------------------------------------")
                generate_sentences(model, 10, index_to_word, word_to_index)
                print("\n")
                sys.stdout.flush()    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sample language model")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=25000)
    parser.add_argument("--model", help="file to store the model")
    parser.add_argument("--dataset", help="input training data file")
    parser.add_argument("--test", action="store_true", default=False, help="True for test, False for train")
    parser.add_argument("--reload", action="store_true", default=False, help="True reload model")
    
    args = parser.parse_args()
    if args.test:
        if os.path.isfile(args.dataset + ".vocab.pkl"):
            vocab = cPickle.load(open(args.dataset + ".vocab.pkl", "r"))
            sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
            index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
            word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
            print "building model..."
            model = GRUTheano(len(index_to_word), hidden_dim=args.hidden_dim, bptt_truncate=-1)
            print "build model done."
            model.load_params(args.model)
            generate_sentences(model, 10, index_to_word, word_to_index)
        else:
            print "[Error]: vocab file is not avaliable!"
    else:
        train(args)


