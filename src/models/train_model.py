from sklearn.feature_extraction.text import TfidfVectorizer
import os

import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
import sys
sys.path.append("/Users/oscarwyatt/data_science/pageview_predictor/src/")
import utils
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import traceback
import warnings
import sys
from sklearn.ensemble import ExtraTreesClassifier
import pickle
from sklearn.preprocessing import normalize
from argparse import ArgumentParser


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

def draw_content_item_distribution_graph(pageview_values, descretizer, view_numbers):
    binned_page_views = discretizer.transform(view_numbers)
    count = {}
    for bin in binned_page_views:
        number = count.get(bin[0], 0)
        count[bin[0]] = number + 1

    plt.plot(binned_page_views, view_numbers, f'r--')
    params = discretizer.get_params()
    plt.title("Distribution of binned regulation content items log10 page views with " + str(params['n_bins']) + " bins and " + params[
        'strategy'] + " strategy")
    plt.ylabel('Log10 of page views')
    plt.xlabel("Bin edges at: " + ", ".join([str(x) for x in discretizer.bin_edges_.tolist()[0]]) + "\n" +  "# items in each bin" + str(count))
    plt.show()

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def generate_corpus_and_vectorise(discretizer, pageview_values):
    corpus = []
    title_corpus = []
    description_corpus = []
    items = os.listdir(utils.content_items_dir())
    taxon_names = []

    count = len(items)
    extra_features = False
    locales = []
    primary_publishing_organisations = []
    schema_types = []
    document_types = []
    y = []

    for i, filename in enumerate(items[0:count]):
        content_item, body, title, description = utils.extract_content_item(filename)
        filename = filename.replace(".json", "").replace("_", "/")
        if content_item != False and filename in pageviews:
            page_views = pageview_values[filename]
            y.append(discretizer.transform(np.array([page_views]).reshape(1, -1))[0][0])
            corpus.append(body)
            title_corpus.append(title)
            description_corpus.append(description)
            # Extra features
            #   Taxon name and count feature
            taxon_name, taxon_count = utils.get_taxon_name_and_count(content_item)
            if taxon_name not in taxon_names:
                taxon_names.append(taxon_name)
            extra_feature = np.zeros((1,15))
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

            extra_feature[0, 13], schema_types  = utils.get_schema_type(schema_types, content_item)

            extra_feature[0, 14], document_types  = utils.get_schema_type(document_types, content_item)

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

    print(X.shape)
    print(extra_features.shape)
    print("Training on " + str(len(y)) + " examples")

    filename = "data/processed/x_shape_" + str(X.shape) + "max_features" + str(max_features) + "extra_features_" + str(len(extra_features))
    x_file = filename + "_X"
    y_file = filename + "_y"

    with open(x_file, 'wb') as fp:
        pickle.dump(X, fp)
    with open(y_file, 'wb') as fp:
        pickle.dump(y, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print("X FILE: " + x_file)
    print("Y FILE: " + y_file)
    return [X, y]

def show_feature_importance(X_extra_features, y):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    forest.fit(X_extra_features, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_extra_features.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_extra_features.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_extra_features.shape[1]), indices)
    plt.xlim([-1, X_extra_features.shape[1]])
    plt.show()

parser = ArgumentParser()
parser.add_argument("-d", "--distribution-graph",
					default=False, help="Draw a graph of the distribution of the content item pageviews")
parser.add_argument("-i", "--feature-importance", default=False, help="Show graph of the importance of various features")
parser.add_argument("-c", "--confusion-matrix", default=False, help="Show confusion matrices during training")

args = vars(parser.parse_args())
pageviews = utils.load_pageviews()
discretizer, view_numbers = utils.generate_discretizer(pageviews)

if args["distribution_graph"]:
    draw_content_item_distribution_graph(pageviews, discretizer, view_numbers)

X = []
y = []
X, y = generate_corpus_and_vectorise(discretizer, pageviews)

if args["feature_importance"]:
    show_feature_importance(X, y)

kf = KFold(n_splits=5)
kf.get_n_splits(X)

scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scores.append(utils.train_and_test_logistic_regression(X_train, y_train, X_test, y_test, args["confusion_matrix"]))
print("Average f1 score :" + str(np.mean(scores)))
model = utils.train_logistic_regression(X, y)
model_filename = "logistic_regression_model.pkl"
pickle.dump(model, open(model_filename, 'wb'))
print("Model saved to " + model_filename)
