require 'CSV'

data = CSV.open("all_guidance_pageview_data.csv").read
data[2...data.length].each do |row|
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
