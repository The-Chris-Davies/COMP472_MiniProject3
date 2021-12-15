#!usr/bin/env python3
#COMP 472 miniproject 3

import pandas as pd
import numpy as np
import gensim.downloader as dl

#loads and returns the dataset
def load_dataset():
    data = pd.read_csv('synonyms.csv')
    return data

#returns the embedding of the word for the given model, or None if the embedding does not exist
def get_word(model, word):
    try:
        return model[word]
    except KeyError:
        return None

#returns true if the word is in the model
def check_in_model(model, word):
    if type(get_word(model, word)) is not type(None):
        return True
    return False

#uses the model to return the closest synonym of the word out of the list of options
def get_synonym(model, word, options):
    reference_word = get_word(model, word)

    #filter options without embeddings
    words = [w for w in options if check_in_model(model, w)]

    #return None if no comparisons can be made
    if (not check_in_model(model, word)) or len(words) == 0:
        return None

    #get similarities of words
    synonyms = list(map(lambda w: model.similarity(word,w), words))

    #otherwise, return the closest word
    return words[synonyms.index(max(synonyms))]

# writes <model_name>-details.csv with the given information and results arra
def write_model_details(model_name, results):
    f = open(model_name + '-details.csv', 'w')
    f.write('question-word,answer-word,guess-word,label\n')
    for line in results:
        f.write('{},{},{},{}\n'.format(line['question'], line['answer'], line['guess'], line['label']))
    f.close()

#performs task 1: evaluation of a model on the dataset
def task_1(model_name):
    # load model
    print('loading model', model_name)
    model = dl.load(model_name)
    print('model loaded')

    print('loading dataset')
    dataset = load_dataset()
    print('dataset loaded')

    # operate model on dataset
    print('finding results')
    results = []
    # data for analysis.csv
    analysis = {'name': model_name, 'size': len(model.index_to_key), 'correct': 0, 'wrong': 0, 'guess': 0}
    for line in dataset.iloc:
        t = True
        guess = get_synonym(model, line['question'], [line['0'], line['1'], line['2'], line['3']])

        #determine label
        label = ''
        if guess == None:
            guess = line['0']
            label = 'guess'
            analysis['guess'] += 1
        elif guess == line.answer:
            label = 'correct'
            analysis['correct'] += 1
        else:
            label = 'wrong'
            analysis['wrong'] += 1

        results.append( {'question': line.question, 'answer': line.answer, 'guess': guess, 'label': label})

    #save results
    write_model_details(model_name, results)

    return analysis

#uses the list of analysis vectors from task_1 to create analysis.csv
def save_analysis(analysis):
    f = open('analysis.csv', 'w')
    f.write('model_name,vocab_size,correct,nonguesses,accuracy\n')
    for analys in analysis:
        f.write('{},{},{},{},{}\n'.format(analys['name'], analys['size'], analys['correct'], analys['correct']+analys['wrong'], analys['correct'] / (analys['correct'] + analys['wrong'])))
    f.close()

# run task_1 and task_2
def main():
    models = ['word2vec-google-news-300', 'glove-wiki-gigaword-200', 'glove-twitter-200', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-300']

    analysis = []
    for model_name in models:
        analysis.append(task_1(model_name))
    save_analysis(analysis)
if __name__ == '__main__':
    main()
