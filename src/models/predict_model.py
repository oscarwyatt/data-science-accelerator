from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import KBinsDiscretizer
import os
import json
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

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
    count[0] = 0
    count[1] = 1
    count[2] = 2
    for bin in binned_page_views:
        count[bin[0]] += 1

    plt.plot(binned_page_views, view_numbers, f'r--')
    params = discretizer.get_params()
    plt.title("Distribution of binned regulation content items with " + str(params['n_bins']) + " bins and " + params[
        'strategy'] + " strategy")
    plt.xlabel('Number of items in each bin' + str(count))
    plt.ylabel("Bin edges at: " + ", ".join([str(x) for x in discretizer.bin_edges_.tolist()[0]]))
    plt.show()


def extract_content_item_body(content_item):
    body_soup = BeautifulSoup(content_item["details"]["body"])
    return body_soup.get_text().replace('\n', '')

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
with open("data/raw/all_guidance_pageview_data.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        pageviews[row['page'].replace("/", "_")[0:200]] = int(row['page_views'])

pageview_values = list(pageviews.values())
pageview_values.sort()
view_numbers = np.array(pageview_values).reshape(-1, 1)
discretizer = KBinsDiscretizer(encode='ordinal', n_bins=3, strategy='kmeans')
discretizer.fit(view_numbers)

if False:
    draw_graph(pageview_values, discretizer, view_numbers)

corpus = []
dir = "data/raw/content_items"
items = os.listdir(dir)
taxon_names = []
taxon_indicies = []
train_n = 10000
c_v_n = 3000
# n = len(items)
taxon_index_column = np.zeros((train_n,1))
y = np.zeros((train_n,1))
for i, filename in enumerate(items[:train_n]):
    file_with_path = dir + "/" + filename
    with open(file_with_path) as file:
        try:
            content_item = json.load(file)
            body = extract_content_item_body(content_item)
            corpus.append(body)
            taxon_name = get_taxon_name(content_item)
            if taxon_name not in taxon_names:
                taxon_names.append(taxon_name)
            taxon_index_column[i, 0] = taxon_names.index(taxon_name)
            page_views = int(pageviews[filename.replace(".json", "")])
            y[i, 0] = discretizer.transform(np.array([page_views]).reshape(1, -1))[0][0]
        except:
            print(filename + " couldn't be loaded or caused JSON error")
            corpus.append("")


vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words=stopwords.words('english') )
X = vectorizer.fit_transform(corpus).toarray()
X = np.append(X, taxon_index_column, axis=1)

reg = LogisticRegression(n_jobs = -1).fit(X, y)
print(reg.score(X, y))
print(reg.coef_)


# Cross validate
taxon_index_column = np.zeros((c_v_n,1))
y = np.zeros((c_v_n,1))
cross_validated_docs = []
c_v_y = np.zeros((c_v_n, 1))
c_v_taxons = np.zeros((c_v_n, 1))
for i, filename in enumerate(items[train_n:train_n + c_v_n]):
    file_with_path = dir + "/" + filename
    with open(file_with_path) as file:
        try:
            content_item = json.load(file)
            cross_validated_docs.append(extract_content_item_body(content_item))
            taxon_name = get_taxon_name(content_item)
            c_v_taxons[i,0] = taxon_names.index(taxon_name)
            page_views = int(pageviews[filename.replace(".json", "")])
            c_v_y[i, 0] = discretizer.transform(np.array([page_views]).reshape(1, -1))[0][0]
        except:
            print("cross validate: " + filename + " couldn't be loaded or caused JSON error")
            cross_validated_docs.append("")

c_v_x = vectorizer.transform(cross_validated_docs).toarray()
c_v_x = np.append(c_v_x, c_v_taxons, axis=1)
predictions = reg.predict(c_v_x)
print("Mean accuracy of cross validation set: " + str(np.mean(predictions == c_v_y)))
