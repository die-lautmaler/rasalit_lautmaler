import csv
import os.path

import spacy

nlp = spacy.load('de_core_news_lg')

# doc = nlp('ich hätte gern, dass du die kugel mal nach rechts bewegst')
verbPOS = ['VERB', 'AUX']
nonVerbPOS = ['DET', 'NOUN', 'PRON', 'X', 'NUM', 'ADJ', 'PART']

with open(os.path.join('data', 'account_start_stuff.csv'), 'r', encoding='utf8') as inf:
    table = csv.DictReader(inf, fieldnames=['text', 'PhraseIntent'], delimiter=';')

    verbSet = set()
    for row in table:
        doc = nlp(row['text'].strip())
        for token in doc:
            if token.pos_ in verbPOS :
                # print(token.text, token.pos_)
                verbSet.add(token.text)

    for x in verbSet:
        print(x)