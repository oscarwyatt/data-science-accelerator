
# This code must be run within a govuk rails app so that the content store can be accessed

require 'CSV'


base_paths = Hash.new(0)
data = CSV.open("all_data.csv").read
data.each do |row|
	begin
		taxons = Services.content_store.content_item(row.first).to_h.dig("links", "taxons")
		if taxons
			base_paths[taxons.last["base_path"]] += 1
		end
	rescue
	p "error"
 	end
end
sorted_base_paths = base_paths.sort_by {|_key, value| value}.reverse.to_h
File.open("sorted_taxon_number_of_most_popular_5000_guidance.csv", 'w') { |file| file.write(sorted_base_paths.to_a.to_csv) }
