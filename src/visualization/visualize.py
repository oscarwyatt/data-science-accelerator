import pandas
from .. import utils

df = pandas.read_csv("data/raw/all_guidance_pageview_data.csv")
df['page_views'].describe()
import csv


# count     92171.000000
# # mean        255.230691
# # std        2309.993873
# # min           0.000000
# # 25%           3.000000
# # 50%          11.000000
# # 75%          54.000000
# # max      210466.000000

log_view_counts = {}
for view in list(df['page_views']):
    if view > 0:
        log = int(math.log10(view))
        count = log_view_counts.get(log, 0)
        log_view_counts[log] = count + 1

print(log_view_counts)

# {0: 44071, 5: 9, 4: 365, 3: 3117, 2: 13263, 1: 31325}


def log_page_views(row):
    num_page_views = int(row['page_views'])
    if num_page_views > 0:
        num_page_views = int(math.log10(num_page_views))
    return num_page_views


# Display time on page vs # of page views
import datetime
import matplotlib.pyplot as plt
import csv

pageviews = []
time_on_page = []
with open("../../data/raw/all_guidance_pageview_data.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        num_page_views = log_page_views(row)
        pageviews.append(num_page_views)
        time = str(row["time_on_page"])
        parts = time.split(':')
        if len(parts) == 3:
            h, m, s = map(int, time.split(':'))
            dur = datetime.timedelta(hours=h, minutes=m, seconds=s)
            time_on_page.append(dur.seconds)
        else:
            time_on_page.append(0)

colors = (0,0,0)
plt.scatter(time_on_page, pageviews, s = 1, c=colors, alpha=0.5)

plt.title("Log10 of guidance pageviews vs mean number of seconds spent on page")
plt.ylabel('Log10 of pageviews')
plt.xlabel("Mean time on page (seconds)")
plt.show()




# Display #links in body vs page views
import csv
import math
import json
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append("/Users/oscarwyatt/data_science/pageview_predictor/src/")
import utils
dir = "data/raw/content_items"
pageviews = []
num_links = []
num_words = []
num_translations = []
count = 0
num_links_for_log_views = {}
num_words_for_log_views = {}
num_translations_for_log_views = {}
with open("data/raw/all_guidance_pageview_data.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        file_with_path = dir + "/" + row['page'].replace("/", "_")[0:200] + ".json"
        with open(file_with_path) as json_file:
            try:
                content_item = json.load(json_file)
            except:
                print(file_with_path + " couldn't be loaded or caused JSON error")
            else:
                num_page_views = int(row['page_views'])
                if num_page_views > 0:
                    num_page_views = int(math.log10(num_page_views))

                # Num links
                link_count = utils.number_of_links(content_item)
                num_links.append(link_count)
                array = num_links_for_log_views.get(num_page_views, [])
                array.append(link_count)
                num_links_for_log_views[num_page_views] = array

                # Num words
                word_count = utils.number_of_words(content_item)
                num_words.append(word_count)
                array = num_words_for_log_views.get(num_page_views, [])
                array.append(word_count)
                num_words_for_log_views[num_page_views] = array

                # Num translations
                translation_count = utils.number_of_translations(content_item)
                num_translations.append(translation_count)
                array = num_translations_for_log_views.get(num_page_views, [])
                array.append(translation_count)
                num_translations_for_log_views[num_page_views] = array

                # Num page views
                pageviews.append(num_page_views)

colors = (0,0,0)
plt.subplot(3, 1, 1)
plt.scatter(num_links, pageviews, s = 1, c=colors, alpha=0.5)
plt.title("Log10 of guidance pageviews vs mean number of links on page")
plt.ylabel('Log10 of pageviews')
plt.xlabel("Number of links on page")

for key, value in num_links_for_log_views.items():
    print("Description for items with log10 of " + str(key) + " pageviews by number of links")
    print(pd.Series(value).describe())

#
# Description for items with log10 of 0 pageviews
# count    43681.000000
# mean         1.062247
# std          4.675216
# min          0.000000
# 25%          0.000000
# 50%          0.000000
# 75%          1.000000
# max        180.000000
# dtype: float64
#
#
# Description for items with log10 of 1 pageviews
# count    31234.000000
# mean         2.192707
# std          9.095742
# min          0.000000
# 25%          0.000000
# 50%          0.000000
# 75%          1.000000
# max        358.000000
# dtype: float64
#
#
# Description for items with log10 of 2 pageviews
# count    13232.000000
# mean         5.168531
# std         26.360280
# min          0.000000
# 25%          0.000000
# 50%          1.000000
# 75%          2.000000
# max       1849.000000
# dtype: float64
#
#
# Description for items with log10 of 3 pageviews
# count    3100.000000
# mean        9.640323
# std        37.145432
# min         0.000000
# 25%         0.000000
# 50%         1.000000
# 75%         5.000000
# max       836.000000
# dtype: float64
#
#
# Description for items with log10 of 4 pageviews
# count    363.000000
# mean       8.680441
# std       32.260376
# min        0.000000
# 25%        0.000000
# 50%        2.000000
# 75%        5.000000
# max      529.000000
# dtype: float64
#
#
# Description for items with log10 of 5 pageviews
# count     9.000000
# mean      4.000000
# std       6.855655
# min       0.000000
# 25%       1.000000
# 50%       2.000000
# 75%       3.000000
# max      22.000000
# dtype: float64


plt.subplot(3, 1, 2)
plt.scatter(num_words, pageviews, s = 1, c=colors, alpha=0.5)

plt.title("Log10 of guidance pageviews vs mean number of words in main body")
plt.ylabel('Log10 of pageviews')
plt.xlabel("Number of words in main body")
plt.show()

for key, value in num_words_for_log_views.items():
    print("Description for items with log10 of " + str(key) + " pageviews by number of words")
    print(pd.Series(value).describe())

#
# Description for items with log10 of 0 pageviews by number of words
# count    43964.000000
# mean       149.809412
# std        626.373577
# min          1.000000
# 25%         26.000000
# 50%         48.000000
# 75%         93.000000
# max      56904.000000
# dtype: float64
#
# Description for items with log10 of 1 pageviews by number of words
# count    31317.000000
# mean       262.159913
# std       1140.887775
# min          1.000000
# 25%         28.000000
# 50%         55.000000
# 75%        120.000000
# max      56904.000000
# dtype: float64
#
# Description for items with log10 of 2 pageviews by number of words
# count    13258.000000
# mean       559.138860
# std       2150.376974
# min          1.000000
# 25%         35.000000
# 50%         70.000000
# 75%        181.750000
# max      56617.000000
# dtype: float64
#
# Description for items with log10 of 3 pageviews by number of words
# count     3111.000000
# mean      1086.615558
# std       3607.589420
# min          1.000000
# 25%         45.000000
# 50%         92.000000
# 75%        443.500000
# max      59470.000000
# dtype: float64
#
# Description for items with log10 of 4 pageviews by number of words
# count       365.000000
# mean       1595.624658
# std       11995.025792
# min           1.000000
# 25%          49.000000
# 50%         109.000000
# 75%         305.000000
# max      221378.000000
# dtype: float64
#
# Description for items with log10 of 5 pageviews by number of words
# count      9.000000
# mean     210.111111
# std      240.174127
# min       11.000000
# 25%       82.000000
# 50%      149.000000
# 75%      243.000000
# max      809.000000
# dtype: float64

plt.subplot(3, 1, 3)
plt.scatter(num_translations, pageviews, s = 1, c=colors, alpha=0.5)

plt.title("Log10 of guidance pageviews vs number of available translations")
plt.ylabel('Log10 of pageviews')
plt.xlabel("Number of translations")
plt.show()

for key, value in num_words_for_log_views.items():
    print("Description for items with log10 of " + str(key) + " pageviews by number of translations")
    print(pd.Series(value).describe())
