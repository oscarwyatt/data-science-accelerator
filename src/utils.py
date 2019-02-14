from bs4 import BeautifulSoup, SoupStrainer
import pickle
import math
import csv
import os
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def extract_content_item_body(content_item):
    try:
        body_soup = BeautifulSoup(content_item["details"]["body"], features='html.parser')
        return body_soup.get_text().replace('\n', '')
    except:
        return ""

def extract_content_item_title(content_item):
    try:
        return str(content_item["title"])
    except:
        return ""

def extract_content_item_description(content_item):
    try:
        return str(content_item["description"])
    except:
        return ""

def number_of_links(content_item):
    try:
        return len(BeautifulSoup(content_item["details"]["body"], features='html.parser', parse_only=SoupStrainer('a')))
    except:
        return 0

def number_of_words(content_item):
    return len(extract_content_item_body(content_item).split(" "))

def number_of_translations(content_item):
    return len(content_item.get("links", {}).get("available_translations", []))

def locale(content_item, locales):
    locale = content_item.get("locale", "")
    if locale not in locales:
        locales.append(locale)
    return [locales.index(locale), locales]

def primary_publishing_organisation(content_item, all_primary_publishing_organisations):
    primary_publishing_organisations = content_item.get("details", {}).get("primary_publishing_organisation", [])
    if any(primary_publishing_organisations):
        primary_publishing_organisation = primary_publishing_organisations[0]["base_path"]
        if primary_publishing_organisation not in all_primary_publishing_organisations:
            all_primary_publishing_organisations.append(primary_publishing_organisation)
        return [all_primary_publishing_organisations.index(primary_publishing_organisation), all_primary_publishing_organisations]
    else:
        return [-1, all_primary_publishing_organisations]

def number_of_documents(content_item):
    return len(content_item.get("links", {}).get("document_collections", []))

def number_of_available_translations(content_item):
    return len(content_item.get("links", {}).get("available_translations", []))

def number_of_organisations(content_item):
    return len(content_item.get("links", {}).get("organisations", []))

def number_of_topics(content_item):
    return len(content_item.get("links", {}).get("topics", []))

def number_of_tags(content_item):
    return len(content_item.get("details", {}).get("tags", []))

def number_of_emphasised_organisations(content_item):
    return len(content_item.get("details", {}).get("emphasised_organisations", []))

def political(content_item):
    return int(content_item.get("details", {}).get("political", False))

def load_pageviews():
    with open(pageviews_pickle_file(), 'rb') as fp:
        pageviews = pickle.load(fp)
        return pageviews

def pageviews_pickle_file():
    return "data/processed/pageviews_data.p"

def generate_pageviews_data():
    pageviews = {}
    with open("data/raw/shuffled_guidance_pageview_data.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            num_page_views = int(row['page_views'])
            if num_page_views > 0:
                num_page_views = int(math.log10(num_page_views))
            pageviews[row['page'].replace("/", "_")[0:200]] = num_page_views
    with open(pageviews_pickle_file, 'wb') as fp:
        pickle.dump(pageviews, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return pageviews

def extract_content_item(filename):
    file_with_path = content_items_dir() + "/" + filename
    saved_processed_file_with_path = "data/processed/content_items/" + filename
    saved_processed_body_file_with_path = "data/processed/content_items_body/" + filename
    content_item = ""
    body = ""
    title = ""
    description = ""
    if os.path.isfile(saved_processed_file_with_path):
        try:
            with open(saved_processed_file_with_path, 'rb') as fp:
                content_item = pickle.load(fp)
            with open(saved_processed_body_file_with_path, 'rb') as fp:
                body = pickle.load(fp)
        except:
            print(filename + " couldn't be loaded or caused JSON error")
            content_item = False

    else:
        with open(file_with_path) as file:
            try:
                content_item = json.load(file)
            except:
                print(filename + " couldn't be loaded or caused JSON error")
                content_item = False
            else:
                with open(saved_processed_file_with_path, 'wb') as fp:
                    pickle.dump(content_item, fp, protocol=pickle.HIGHEST_PROTOCOL)
                body = extract_content_item_body(content_item)
                with open(saved_processed_body_file_with_path, 'wb') as fp:
                    pickle.dump(body, fp, protocol=pickle.HIGHEST_PROTOCOL)

    title = extract_content_item_title(content_item)
    description = extract_content_item_description(content_item)
    return [content_item, body, title, description]


def content_items_dir():
    return "data/raw/content_items"

def generate_discretizer(pageviews):
    pageview_values = list(pageviews.values())
    sorted_pageview_values = pageview_values
    sorted_pageview_values.sort()
    view_numbers = np.array(sorted_pageview_values).reshape(-1, 1)
    discretizer = KBinsDiscretizer(encode='ordinal', n_bins=3, strategy='kmeans')
    discretizer.fit(view_numbers)
    return [discretizer, view_numbers]


def get_taxon_name_and_count(content_item):
    parent_taxon = content_item.get("links", {}).get("taxons", {})
    if any(parent_taxon):
        parent_taxon = parent_taxon[0]
        count = 0
        while parent_taxon != {}:
            possible_parent_taxon = parent_taxon.get("links", {}).get("parent_taxons", {})
            if possible_parent_taxon != {}:
                parent_taxon = possible_parent_taxon[0]
                count += 1
            else:
                break
        return [parent_taxon["base_path"], count]
    else:
        return ["", 0]

def train_and_test_logistic_regression(X_train, y_train, X_test, y_test, show_cf=False):
    reg = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=200).fit(X_train, y_train)
    pred = reg.predict(X_test)
    if show_cf:
        show_confusion_matrix(y_test, pred)
    return f1_score(y_test, pred, average='micro')

def show_confusion_matrix(y_true, y_pred):
    cnf_matrix = confusion_matrix(y_true, y_pred)

    print(cnf_matrix)
    plt.figure()
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()

    fmt = 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



