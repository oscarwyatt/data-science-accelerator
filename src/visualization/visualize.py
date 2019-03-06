import pandas
# from .. import utils
# import csv
import math
import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/Users/oscarwyatt/data_science/pageview_predictor/src/")
import utils
import os


raw_pageviews = utils.load_raw_pageviews()
pageviews_dataframe = pandas.DataFrame(list(raw_pageviews.values()))
print(pageviews_dataframe.describe())

# count     92171.000000
# mean        255.230691
# std        2309.993873
# min           0.000000
# 25%           3.000000
# 50%          11.000000
# 75%          54.000000
# max      210466.000000

log_view_counts = {}
for view in list(pageviews_dataframe):
    if view > 0:
        log = (math.log10(view)).floor
        count = log_view_counts.get(log, 0)
        log_view_counts[log] = count + 1

print(log_view_counts)
# {0: 44071, 5: 9, 4: 365, 3: 3117, 2: 13263, 1: 31325}


# Display number of page links in body vs page views
all_pageviews = []
num_links = []
num_words = []
num_translations = []
count = 0
num_links_for_log_views = {}
num_words_for_log_views = {}
num_translations_for_log_views = {}
dir = utils.content_items_dir()
items = os.listdir(utils.content_items_dir())
for filename in items:
    content_item, body, title, description = utils.extract_content_item(filename)
    filename = filename.replace(".json", "").replace("_", "/")
    if content_item != False and filename in raw_pageviews:
        pageviews = raw_pageviews[filename]

        # Num links
        link_count = utils.number_of_links(content_item)
        num_links.append(link_count)
        array = num_links_for_log_views.get(pageviews, [])
        array.append(link_count)
        num_links_for_log_views[pageviews] = array

        # Num words
        word_count = utils.number_of_words(content_item)
        num_words.append(word_count)
        array = num_words_for_log_views.get(pageviews, [])
        array.append(word_count)
        num_words_for_log_views[pageviews] = array

        # Num translations
        translation_count = utils.number_of_translations(content_item)
        num_translations.append(translation_count)
        array = num_translations_for_log_views.get(pageviews, [])
        array.append(translation_count)
        num_translations_for_log_views[pageviews] = array

        # Num page views
        all_pageviews.append(pageviews)

colors = (0,0,0)
plt.subplot(3, 1, 1)
plt.scatter(num_links, all_pageviews, s = 1, c=colors, alpha=0.5)
plt.title("Log10 of guidance pageviews vs mean number of links on page")
plt.ylabel('Log10 of pageviews')
plt.xlabel("Number of links on page")

for key, value in num_links_for_log_views.items():
    print("Description for items with log10 of " + str(key) + " pageviews by number of links")
    print(pd.Series(value).describe())

plt.subplot(3, 1, 2)
plt.scatter(num_words, all_pageviews, s = 1, c=colors, alpha=0.5)

plt.title("Log10 of guidance pageviews vs mean number of words in main body")
plt.ylabel('Log10 of pageviews')
plt.xlabel("Number of words in main body")
plt.show()

for key, value in num_words_for_log_views.items():
    print("Description for items with log10 of " + str(key) + " pageviews by number of words")
    print(pd.Series(value).describe())

plt.subplot(3, 1, 3)
plt.scatter(num_translations, all_pageviews, s = 1, c=colors, alpha=0.5)

plt.title("Log10 of guidance pageviews vs number of available translations")
plt.ylabel('Log10 of pageviews')
plt.xlabel("Number of translations")
plt.show()

for key, value in num_words_for_log_views.items():
    print("Description for items with log10 of " + str(key) + " pageviews by number of translations")
    print(pd.Series(value).describe())







# Display time on page vs # of page views
# Archived as time on page is not something we can consider for new items

# def log_page_views(row):
#     num_page_views = int(row['page_views'])
#     if num_page_views > 0:
#         num_page_views = (math.log10(num_page_views)).floor
#     return num_page_views

# import datetime
# import matplotlib.pyplot as plt
# import csv
#
# pageviews = []
# time_on_page = []
# with open("../../data/raw/all_guidance_pageview_data.csv") as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         num_page_views = log_page_views(row)
#         pageviews.append(num_page_views)
#         time = str(row["time_on_page"])
#         parts = time.split(':')
#         if len(parts) == 3:
#             h, m, s = map(int, time.split(':'))
#             dur = datetime.timedelta(hours=h, minutes=m, seconds=s)
#             time_on_page.append(dur.seconds)
#         else:
#             time_on_page.append(0)
#
# colors = (0,0,0)
# plt.scatter(time_on_page, pageviews, s = 1, c=colors, alpha=0.5)
#
# plt.title("Log10 of guidance pageviews vs mean number of seconds spent on page")
# plt.ylabel('Log10 of pageviews')
# plt.xlabel("Mean time on page (seconds)")
# plt.show()