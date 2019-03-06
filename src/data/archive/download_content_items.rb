# Run this in a console in some rails app OTHER than content store
# expects input file generated from manually downloading GA pageview data
# as a csv from a report much like:
# https://analytics.google.com/analytics/web/#/report/content-pages/a26179049w50705554p53872948/explorer-table.plotKeys=%5B%5D&explorer-table.rowStart=0&explorer-table.rowCount=5000
# You can optionally concatenate these files with src/data/archive/concatenate_downloads_of_ga_pageview_csv_files.rb

require 'CSV'

data = CSV.open("all_guidance_pageview_data.csv").read
data[2...data.length].each do |row| # Don't include header
    url = row.first
    begin
        filename = url.gsub("/", "_")[0...200] + ".json"
        File.open("content_items/#{filename}","w") do |f|
            f.write(Services.content_store.content_item(url).to_h.to_json)
        end
    rescue
        p "couldn't download: #{url}"
    end
end
