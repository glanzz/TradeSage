import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence


def load_conversations(hparams, data):
    # dictionary of line id to text
    inputs, outputs = [], []
    for index, row in data.iterrows():
        question, answer = row["Question"], row["Answer"]
        inputs.append(preprocess_sentence(question))
        outputs.append(preprocess_sentence(answer))
        if len(inputs) >= hparams.max_samples:
            return inputs, outputs
            
    return inputs, outputs


def tokenize_and_filter(hparams, tokenizer, questions, answers):
    tokenized_questions, tokenized_answers = [], []

    for (question, answer) in zip(questions, answers):
        # tokenize sentence
        sentence1 = hparams.start_token + tokenizer.encode(question) + hparams.end_token
        sentence2 = hparams.start_token + tokenizer.encode(answer) + hparams.end_token

        # check tokenize sentence length
        if (
            len(sentence1) <= hparams.max_length
            and len(sentence2) <= hparams.max_length
        ):
            tokenized_questions.append(sentence1)
            tokenized_answers.append(sentence2)

    # pad tokenized sentences
    tokenized_questions = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_questions, maxlen=hparams.max_length, padding="post"
    )
    tokenized_answers = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_answers, maxlen=hparams.max_length, padding="post"
    )

    return tokenized_questions, tokenized_answers


def get_dataset(hparams):
    # download corpus
    import pandas as pd
    # filenames = ["philosophy.csv", "timing.csv", "psychology.csv"]
    base = "data/"
    # df1 = pd.read_csv(base + "data2/philosophy.csv")
    # df2 = pd.read_csv(base + "data2/timing.csv")
    # df3 = pd.read_csv(base + "data2/psychology.csv")
    # df4 = pd.read_csv(base + "data2/personal.csv")
    # df5 = pd.read_csv(base + "data2/strategy.csv")
    # df6 = pd.read_csv(base + "data2/george_soros_accum.csv")


    # df7 = pd.read_csv(base + "data/philosophy.csv")
    # df9 = pd.read_csv(base + "data/psychology.csv")
    # df11 = pd.read_csv(base + "data/strategy.csv")
    df8 = pd.read_csv(base + "qnagen.csv")

    # Concatenate all DataFrames
    # final_data = pd.concat([df1, df2, df3, df4, df5, df6, df7, df9, df11], ignore_index=True)
    final_data = pd.concat([df8], ignore_index=True)

    questions, answers = load_conversations(
        hparams, final_data
    )

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2**13
    )

    hparams.start_token = [tokenizer.vocab_size]
    hparams.end_token = [tokenizer.vocab_size + 1]
    hparams.vocab_size = tokenizer.vocab_size + 2

    questions, answers = tokenize_and_filter(hparams, tokenizer, questions, answers)

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"inputs": questions, "dec_inputs": answers[:, :-1]}, answers[:, 1:])
    )
    dataset = dataset.cache()
    dataset = dataset.shuffle(len(questions))
    dataset = dataset.batch(hparams.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print(f"********* VOCAB_SIZE: {hparams.vocab_size}******************")

    return dataset, tokenizer
