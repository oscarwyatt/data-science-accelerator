# This is able to use Google BigQuery to download pageview data for all pages
# on GOV.UK
# You'll need a Google BigQuery private key (not included in repo for obvious reasons)


from datetime import datetime, timedelta
import pandas as pd
import pickle
import math

project_id = 'govuk-bigquery-analytics'
key_path = "/Users/oscarwyatt/bigquery.json"

d = datetime.today() - timedelta(days=30)
delta = timedelta(days=1)
raw_pageviews = {}
while d <= datetime.today() - timedelta(days=1):
    query = """SELECT
      pagepath,
      COUNT(*) AS pageviews,
      COUNT(DISTINCT session_id) AS unique_pageviews
    FROM (
      SELECT
        hits.page.pagePath,
        hits.type AS hitType,
        CONCAT(fullVisitorId, CAST(visitStartTime AS STRING)) AS session_id
      FROM
        `govuk-bigquery-analytics.87773428.ga_sessions_TIMESTAMP` AS sessions
      CROSS JOIN
        UNNEST(sessions.hits) AS hits)
    WHERE
      hitType = 'PAGE'
    GROUP BY
      pagePath
    ORDER BY
      pageviews DESC""".replace("TIMESTAMP", d.strftime("%Y%m%d"))

    df_in = pd.io.gbq.read_gbq(query,
                               project_id=project_id,
                               reauth=False,
                               verbose=True,
                               private_key=key_path,
                               dialect="standard")
    for index, row in df_in.iterrows():
        unique_pageviews = raw_pageviews.get(row['pagepath'], 0)
        unique_pageviews += row['unique_pageviews']
        raw_pageviews[row['pagepath']] = unique_pageviews
    print("Completed for " +  d.strftime("%Y%m%d"))
    d += delta

with open("data/raw/raw_pageviews.pkl", 'wb') as fp:
    pickle.dump(raw_pageviews, fp)

processed_pageviews = {}
for pagepath, pageviews in raw_pageviews.items():
    processed_pageviews[pagepath] = math.floor(math.log10(pageviews))

with open("data/processed/processed_pageviews.pkl", 'wb') as fp:
    pickle.dump(processed_pageviews, fp)