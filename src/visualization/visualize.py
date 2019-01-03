import pandas
df = pandas.read_csv("data/raw/all_guidance_pageview_data.csv")
df['page_views'].describe()


# count     92171.000000
# # mean        255.230691
# # std        2309.993873
# # min           0.000000
# # 25%           3.000000
# # 50%          11.000000
# # 75%          54.000000
# # max      210466.000000