from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import KBinsDiscretizer
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import csv
import nltk
from nltk.stem.porter import PorterStemmer
import math
import sys
sys.path.append("/Users/oscarwyatt/data_science/pageview_predictor/src/")
import utils


def get_taxon_name(content_item):
    parent_taxon = content_item.get("links", {}).get("taxons", {})
    if any(parent_taxon):
        parent_taxon = parent_taxon[0]
        while parent_taxon != {}:
            possible_parent_taxon = parent_taxon.get("links", {}).get("parent_taxons", {})
            if possible_parent_taxon != {}:
                parent_taxon = possible_parent_taxon[0]
            else:
                break
        return parent_taxon["base_path"]
    else:
        return ""


def draw_graph(pageview_values, descretizer, view_numbers):
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

pageviews = {}
with open("data/raw/shuffled_guidance_pageview_data.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        num_page_views = int(row['page_views'])
        if num_page_views > 0:
            num_page_views = int(math.log10(num_page_views))
        pageviews[row['page'].replace("/", "_")[0:200]] = num_page_views

pageview_values = list(pageviews.values())
sorted_pageview_values = pageview_values
sorted_pageview_values.sort()
view_numbers = np.array(sorted_pageview_values).reshape(-1, 1)
discretizer = KBinsDiscretizer(encode='ordinal', n_bins=5, strategy='uniform')
discretizer.fit(view_numbers)


if False:
    draw_graph(pageview_values, discretizer, view_numbers)

# Try and save memory
view_numbers = []
sorted_pageview_values = []

corpus = []
dir = "data/raw/content_items"
items = os.listdir(dir)
taxon_names = []
taxon_indicies = []
c_v_n = int(float(len(items)) * 0.3)
train_n = len(items) - c_v_n
#
c_v_n = 3000
train_n = 7000

extra_features = np.zeros((train_n,3))
y = np.zeros((train_n,1))
for i, filename in enumerate(items[:train_n]):
    file_with_path = dir + "/" + filename
    with open(file_with_path) as file:
        try:
            content_item = json.load(file)
        except:
            print(filename + " couldn't be loaded or caused JSON error")
            corpus.append("")
            extra_features[i, 0] = -1
            extra_features[i, 1] = -1
            extra_features[i, 2] = -1
            y[i, 0] = -1
        else:
            corpus.append(utils.extract_content_item_body(content_item))

            # Extra features
            #   Taxon name feature
            taxon_name = get_taxon_name(content_item)
            if taxon_name not in taxon_names:
                taxon_names.append(taxon_name)
            extra_features[i, 0] = taxon_names.index(taxon_name)
            #   Number of links feature
            extra_features[i, 1] = utils.number_of_links(content_item)
            # Number of words feature
            extra_features[i,2] = utils.number_of_words(content_item)

            # Add to y array
            page_views = pageviews[filename.replace(".json", "")]
            y[i, 0] = discretizer.transform(np.array([page_views]).reshape(1, -1))[0][0]



vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english', max_features=50000 )
X = vectorizer.fit_transform(corpus).toarray()
corpus = []
X = np.append(X, extra_features, axis=1)
extra_features = []

reg = LogisticRegression().fit(X, y)
print(reg.coef_)
print(X.shape)
x = []


# Cross validate
taxon_index_column = np.zeros((c_v_n - 1,1))
y = np.zeros((c_v_n - 1,1))
cross_validated_docs = []
c_v_y = np.zeros((c_v_n - 1,1))
extra_features = np.zeros((c_v_n - 1,3))
start_index = train_n + 1
for i, filename in enumerate(items[start_index:train_n + c_v_n]):
    file_with_path = dir + "/" + filename
    with open(file_with_path) as file:
        try:
            content_item = json.load(file)
        except:
            print("cross validate: " + filename + " couldn't be loaded or caused JSON error")
            cross_validated_docs.append("")
            c_v_y[i, 0] = -1
        else:
            cross_validated_docs.append(utils.extract_content_item_body(content_item))

            taxon_name = get_taxon_name(content_item)
            extra_features[i,0] = taxon_names.index(taxon_name)
            extra_features[i,1] = utils.number_of_links(content_item)
            extra_features[i,2] = utils.number_of_words(content_item)
            page_views = int(pageviews[filename.replace(".json", "")])
            c_v_y[i, 0] = discretizer.transform(np.array([page_views]).reshape(1, -1))[0][0]


c_v_x = vectorizer.transform(cross_validated_docs)
c_v_x = np.append(c_v_x.toarray(), extra_features, axis=1)
predictions = reg.predict(c_v_x)
# Ignore items that weren't loaded when making accuracy
for i, y in enumerate(c_v_y):
    if y == -1:
        predictions[i] = y
print("Mean accuracy of cross validation set: " + str(np.mean(predictions == c_v_y)))
print(reg.score(predictions, c_v_y))