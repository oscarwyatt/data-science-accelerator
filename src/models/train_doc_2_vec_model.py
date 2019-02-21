#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import sys
sys.path.append("/Users/oscarwyatt/data_science/pageview_predictor/src/")
import utils
import os
import nltk
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils as sklearnutils
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def model(tagged_words):
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=8)
    model_dbow.build_vocab([x for x in tqdm(tagged_words)])
    for epoch in range(30):
        model_dbow.train(sklearnutils.shuffle([x for x in tqdm(tagged_words)]), total_examples=len(tagged_words), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha
    return model_dbow


def vec_for_learning(model, tagged_docs):
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in tagged_docs])
    return targets, regressors

def fetch_tagged_data(X, y):
    tagged_data = []
    for i, tokenized_body in enumerate(X):
        tagged_data.append(TaggedDocument(words=tokenized_body, tags=[y[i]]))
    return tagged_data

corpus = []
title_corpus = []
description_corpus = []
dir = utils.content_items_dir()
items = os.listdir(dir)
taxon_names = []

pageviews = utils.load_pageviews()
count = len(pageviews)
extra_features = False
locales = []
primary_publishing_organisations = []
y = []
discretizer, view_numbers = utils.generate_discretizer(pageviews)

data = []


for i, filename in enumerate(items[0:count]):
    content_item, body, title, description = utils.extract_content_item(filename)
    if content_item != False:
        filename = filename.replace(".json", "")
        example_y = -1
        if filename in pageviews:
            page_views = pageviews[filename]
            example_y = discretizer.transform(np.array([page_views]).reshape(1, -1))[0]
        else:
            next()
        data.append(tokenize_text(body))
        y.append(example_y[0])
        taxon_name, taxon_count = utils.get_taxon_name_and_count(content_item)
        if taxon_name not in taxon_names:
            taxon_names.append(taxon_name)
        extra_feature = np.zeros((1,13))
        extra_feature[0, 0] = taxon_names.index(taxon_name)
        extra_feature[0, 1] = taxon_count
        #   Number of links feature
        extra_feature[0, 2] = utils.number_of_links(content_item)
        # Number of words feature
        extra_feature[0, 3] = utils.number_of_words(content_item)
        # Locale feature
        extra_feature[0, 4], locales = utils.locale(content_item, locales)
        # Primary publishing organisation
        extra_feature[0, 5], primary_publishing_organisations = utils.primary_publishing_organisation(content_item, primary_publishing_organisations)
        # Number of documents
        extra_feature[0, 6] = utils.number_of_documents(content_item)

        extra_feature[0, 7] = utils.number_of_available_translations(content_item)

        extra_feature[0, 8] = utils.number_of_organisations(content_item)

        extra_feature[0, 9] = utils.number_of_topics(content_item)

        extra_feature[0, 10] = utils.number_of_tags(content_item)

        extra_feature[0, 11] = utils.number_of_emphasised_organisations(content_item)

        extra_feature[0, 12] = utils.political(content_item)
        if isinstance(extra_features, bool):
            extra_features = extra_feature
        else:
            extra_features = np.append(extra_features, extra_feature, axis=0)

y = np.asarray(y).transpose()
extra_features = normalize(extra_features, axis=0, norm='max')

kf = KFold(n_splits=3)
kf.get_n_splits(len(data))

scores = []

data = np.asarray(data)

for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = y[train_index], y[test_index]

    tagged_train_data = fetch_tagged_data(X_train, y_train)
    tagged_test_data = fetch_tagged_data(X_test, y_test)

    train_model = model(tagged_train_data)
    test_model = model(tagged_test_data)

    y_train, X_train = vec_for_learning(train_model, tagged_train_data)
    y_test, X_test = vec_for_learning(test_model, tagged_test_data)

    X_train = np.asarray(X_train)
    train_extra_features = np.asarray(extra_features[train_index])
    X_test = np.asarray(X_test)
    test_extra_features = np.asarray(extra_features[test_index])
    X_test = np.append(X_test, test_extra_features, axis=1)
    X_train = np.append(X_train, train_extra_features, axis=1)
    scores.append(utils.train_and_test_logistic_regression(X_train, y_train, X_test, y_test))

print("Average f1 score :" + str(np.mean(scores)))
