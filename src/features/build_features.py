from argparse import ArgumentParser
import sys
sys.path.append("/Users/oscarwyatt/data_science/pageview_predictor/src/")
import utils
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn import utils as sklearnutils
from sklearn.preprocessing import normalize

def discretize(discretizer, page_views):
    page_views_class = discretizer.transform(np.array([page_views]).reshape(1, -1))[0][0]
    labels = { 0: "low", 1: "medium", 2: "high" }
    return labels[int(page_views_class)]

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

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

def build_extra_features(content_item, taxon_names, locales, primary_publishing_organisations, schema_types, document_types):
    taxon_name, taxon_count = utils.get_taxon_name_and_count(content_item)
    if taxon_name not in taxon_names:
        taxon_names.append(taxon_name)
    extra_feature = np.zeros((1,15))
    extra_feature[0, 0] = taxon_names.index(taxon_name)
    extra_feature[0, 1] = taxon_count
    extra_feature[0, 2] = utils.number_of_links(content_item)
    extra_feature[0, 3] = utils.number_of_words(content_item)
    extra_feature[0, 4], locales = utils.locale(content_item, locales)
    extra_feature[0, 5], primary_publishing_organisations = utils.primary_publishing_organisation(content_item, primary_publishing_organisations)
    extra_feature[0, 6] = utils.number_of_documents(content_item)
    extra_feature[0, 7] = utils.number_of_available_translations(content_item)
    extra_feature[0, 8] = utils.number_of_organisations(content_item)
    extra_feature[0, 9] = utils.number_of_topics(content_item)
    extra_feature[0, 10] = utils.number_of_tags(content_item)
    extra_feature[0, 11] = utils.number_of_emphasised_organisations(content_item)
    extra_feature[0, 12] = utils.political(content_item)
    extra_feature[0, 13], schema_types  = utils.get_schema_type(schema_types, content_item)
    extra_feature[0, 14], document_types  = utils.get_schema_type(document_types, content_item)
    return [extra_feature, taxon_names, locales, primary_publishing_organisations, schema_types, document_types]

def build_vectorize():
    corpus = []
    title_corpus = []
    description_corpus = []
    items = os.listdir(utils.content_items_dir())
    taxon_names = []

    count = 100#len(items)
    extra_features = False
    locales = []
    primary_publishing_organisations = []
    schema_types = []
    document_types = []
    y = []

    pageviews = utils.load_pageviews()
    discretizer, view_numbers = utils.generate_discretizer(pageviews)

    for i, filename in enumerate(items[0:count]):
        content_item, body, title, description = utils.extract_content_item(filename)
        filename = filename.replace(".json", "").replace("_", "/")
        if content_item != False and filename in pageviews:
            page_views = pageviews[filename]
            y.append(discretize(discretizer, page_views))
            corpus.append(body)
            title_corpus.append(title)
            description_corpus.append(description)
            extra_feature, taxon_names, locales, primary_publishing_organisations, schema_types, document_types = build_extra_features(content_item, taxon_names, locales, primary_publishing_organisations, schema_types, document_types)
            if isinstance(extra_features, bool):
                extra_features = extra_feature
            else:
                extra_features = np.append(extra_features, extra_feature, axis=0)

    y = np.asarray(y)
    max_features = 500
    vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english', max_features=max_features )
    X = vectorizer.fit_transform(corpus).toarray()
    corpus = []

    title_vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english', max_features=max_features )
    title_X = title_vectorizer.fit_transform(title_corpus).toarray()
    X = np.append(X, title_X, axis=1)
    title_X = []

    description_vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english', max_features=max_features )
    description_X = description_vectorizer.fit_transform(description_corpus).toarray()
    X = np.append(X, description_X, axis=1)
    description_X = []

    extra_features = normalize(extra_features, axis=0, norm='max')
    X = np.append(X, extra_features, axis=1)

    print(str(len(y)) + " examples")

    filename = "data/processed/vectorize"
    x_file = filename + "_X"
    y_file = filename + "_y"

    with open(x_file, 'wb') as fp:
        pickle.dump(X, fp)
    with open(y_file, 'wb') as fp:
        pickle.dump(y, fp)

    print("X FILE: " + x_file)
    print("Y FILE: " + y_file)

def build_doc_to_vec():
    corpus = []
    dir = utils.content_items_dir()
    items = os.listdir(dir)
    taxon_names = []

    pageviews = utils.load_pageviews()
    count = 1000#len(pageviews)
    extra_features = False
    locales = []
    primary_publishing_organisations = []
    schema_types = []
    document_types = []
    y = []
    discretizer, view_numbers = utils.generate_discretizer(pageviews)

    for i, filename in enumerate(items[0:count]):
        content_item, body, title, description = utils.extract_content_item(filename)
        filename = filename.replace(".json", "").replace("_", "/")
        if content_item != False and filename in pageviews:
            page_views = pageviews[filename]
            corpus.append(tokenize(body) + tokenize(title) + tokenize(description))
            y.append(discretize(discretizer, page_views))
            extra_feature, taxon_names, locales, primary_publishing_organisations, schema_types, document_types = build_extra_features(content_item, taxon_names, locales, primary_publishing_organisations, schema_types, document_types)
            if isinstance(extra_features, bool):
                extra_features = extra_feature
            else:
                extra_features = np.append(extra_features, extra_feature, axis=0)

    y = np.asarray(y).transpose()
    extra_features = normalize(extra_features, axis=0, norm='max')
    corpus = np.asarray(corpus)

    tagged_data = []
    for i, tokenized_body in enumerate(corpus):
        tagged_data.append(TaggedDocument(words=tokenized_body, tags=[y[i]]))

    tagged_data_model = model(tagged_data)
    y, X = vec_for_learning(tagged_data_model, tagged_data)
    X = np.asarray(X)
    X = np.append(X, extra_features, axis=1)

    print(str(len(y)) + " examples")

    filename = "data/processed/doc_to_vec"
    x_file = filename + "_X"
    y_file = filename + "_y"

    with open(x_file, 'wb') as fp:
        pickle.dump(X, fp)
    with open(y_file, 'wb') as fp:
        pickle.dump(y, fp)

    print("X FILE: " + x_file)
    print("Y FILE: " + y_file)

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        default="logistic_regression",
                        help="Choose between 'vectorize' and 'doc_to_vec'")
    args = vars(parser.parse_args())

    if args["model"] == "vectorize":
        build_vectorize()
    else:
        build_doc_to_vec()

if __name__== "__main__":
    main()