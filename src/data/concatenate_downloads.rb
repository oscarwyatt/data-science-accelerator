# Concatenate downloads from GA


require 'CSV'
output_filename = "all_data.csv"
highest = 0
highest_row = ""
CSV.open(output_filename, "w") do |csv|
    csv << %w(page page_views unique_page_views time_on_page entrances bounce exit page_value)
    Dir["*.csv"].each do |filename|
        if filename != output_filename
            p filename
            file = File.open(filename).read
            rows = file.split("\n")
            rows.each do |row|
                row = row.split("\"")
                if row.any?
                    last = row.pop
                    subrow = last.split(",")
                    row += subrow.delete_if{|entry| entry.to_s.length == 0 }

                    row = row.delete_if {|entry| entry.gsub(",", "").gsub(" ", "").length == 0 }
                    no_nil_columns = row.select { |column| column.to_s.length == 0 }.any?
                    if row.count >= 8 and (not no_nil_columns)
                        row = row.map{|entry| entry.gsub(",", "")}
                        row[1] = row[1].gsub(",", "").to_i
                        row[2] = row[2].gsub(",", "").to_i
                        row[4] = row[4].gsub(",", "").to_i

                        while row.count > 8
                            row.pop
                        end

                        csv << row
                        if row[1] > highest
                            highest = row[1]
                            highest_row = row
                        end
                    end
                end
            end
        end
    end
end