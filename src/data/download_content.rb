# Simply follow these steps and all live content items will be saved to a folder as separate json files

# 1. `cd ~/govuk/govuk-puppet/development-vm`
# 2. `vagrant up && vagrant ssh`
# 3. `cd /var/govuk/content-store`
# 4. `rails c`

count = 0
ContentItem.all.each do |content_item|
    if content_item["phase"] == "live"
        count += 1
        filename = content_item["base_path"].gsub("/", "_")[0...200] + ".json"
        File.open("content_items/#{filename}","w") do |f|
          f.write(ContentItemPresenter.new(content_item, content_item["base_path"]).to_json)
        end
    end
end
puts "Saved #{count} content items to #{Dir.pwd + "/content_items"}"