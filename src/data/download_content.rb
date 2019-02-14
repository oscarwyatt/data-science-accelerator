require 'CSV'

# From some other app with filenames in csv
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


# From within content store
ContentItem.all.each do |content_item|
	filename = content_item["base_path"].gsub("/", "_")[0...200] + ".json"
	File.open("content_items/#{filename}","w") do |f|
	  f.write(ContentItemPresenter.new(content_item, api_url_method).to_json)
	end
end