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
import json
import math

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

def extract_content_item_schema_name(content_item):
    try:
        return content_item['schema_name']
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
    return "data/processed/processed_pageviews.pkl"

def load_raw_pageviews():
    with open(raw_pageviews_pickle_file(), 'rb') as fp:
        pageviews = pickle.load(fp)
        return pageviews

def raw_pageviews_pickle_file():
    return "data/raw/raw_pageviews.pkl"

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
        try:
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
        except:
            print(file_with_path + " doesn't exist")
    schema = extract_content_item_schema_name(content_item)
    if schema not in schema_types():
        return [False, "", "", ""]
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
    discretizer = KBinsDiscretizer(encode='ordinal', n_bins=number_bins(), strategy='kmeans')
    discretizer.fit(view_numbers)
    return [discretizer, view_numbers]

def number_bins()
    return 3

def process_view_numbers_for_page(views):
    return math.floor(math.log10(views))

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

def get_schema_type(schema_types, content_item):
    schema = content_item['schema_name']
    if schema not in schema_types:
        schema_types.append(schema)
    return [schema_types.index(schema), schema_types]

def get_document_type(document_types, content_item):
    document_type = content_item['document_type']
    if document_type not in document_types:
        document_types.append(document_type)
    return [document_types.index(document_type), document_types]

def train_and_test_logistic_regression(X_train, y_train, X_test, y_test, show_cf=False):
    reg = train_logistic_regression(X_train, y_train)
    pred = reg.predict(X_test)
    confusion_matrix = np.zeros((utils.number_bins(),utils.number_bins()))
    if show_cf:
        confusion_matrix = show_confusion_matrix(y_test, pred)
    return [f1_score(y_test, pred, average='micro'), confusion_matrix]

def train_logistic_regression(X_train, y_train):
    return LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=200).fit(X_train, y_train)

def show_confusion_matrix(y_true, y_pred):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix)
    return cnf_matrix

def plot_confusion_matrix(cnf_matrix):
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
    return cnf_matrix

def schema_types():
    return ['answer', 'calendar', 'case_study', 'consultation', 'contact', 'corporate_information_page', 'detailed_guide', 'document_collection', 'email_alert_signup', 'finder', 'finder_email_signup', 'generic', 'generic_with_external_related_links', 'guide', 'help_page', 'hmrc_manual', 'hmrc_manual_section', 'html_publication', 'knowledge_alpha', 'licence', 'local_transaction', 'manual', 'manual_section', 'news_article', 'organisation', 'person', 'place', 'publication', 'role', 'role_appointment', 'simple_smart_answer', 'specialist_document', 'speech', 'statistical_data_set', 'statistics_announcement', 'take_part', 'taxon', 'topic', 'topical_event_about_page', 'transaction', 'travel_advice', 'working_group', 'world_location', 'world_location_news_article']