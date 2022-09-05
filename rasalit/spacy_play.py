import spacy

nlp = spacy.load('de_core_news_lg')

doc = nlp('ich hätte gern, dass du die kugel mal nach rechts bewegst')

print(doc[1], doc[1].lemma_, doc[1].pos_)

print('----')

for chunk in doc.noun_chunks:
    print(chunk.text, '->', chunk.lemma_)

print('----')

for token in doc:
    print(token.text, token.dep_, token.pos_)

print('pipeline', nlp.pipe_names)
# print(nlp.__dict__)