from datetime import datetime, timedelta
import pandas as pd
import pickle

project_id = 'govuk-bigquery-analytics'
key_path = "/Users/oscarwyatt/bigquery.json"

d = datetime.today() - timedelta(days=30)
delta = timedelta(days=1)
results = {}
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
        unique_pageviews = results.get(row['pagepath'], 0)
        unique_pageviews += row['unique_pageviews']
        results[row['pagepath']] = unique_pageviews

    d += delta

with open("unique_pageviews.pkl", 'wb') as fp:
    pickle.dump(results, fp)
#
rawdata = {}
with open("unique_pageviews.pkl", "rb") as f:
    rawdata = f.read()

myobj = pickle.loads(rawdata)