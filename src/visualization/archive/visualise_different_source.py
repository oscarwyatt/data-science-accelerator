# Plots GA data showing the kind of referral that a page gets against the number of visits for that referral type


import matplotlib.pyplot as plt
import os
import csv

definitions = ["path", "date", "range", "type", "segment", "bounce", "exit", "sessions", "time_on_page", "page_views"]

referral = []
organic = []
direct = []
email = []
totals = []
colours = ['r', 'g', 'b', 'k', 'y']

file_count = 0
files = os.listdir()
for filename in files:
    if ".csv" in filename:
        with open(filename) as file:
            referral = []
            organic = []
            direct = []
            email = []
            totals = []
            csv_reader = csv.reader(file, delimiter=',')
            line_count = 0
            segment_count = 1
            total = 0
            for row in csv_reader:
                if line_count >= 7:
                    segment = row[3]
                    # print(row[8])
                    data = int(row[8].replace(',', ''))
                    if segment == 'Referral':
                        referral.append(data)
                        totals.append(total)
                        total = 0
                    elif segment == 'Organic':
                        organic.append(data)
                    elif segment == 'Direct/none':
                        # direct = relative_append(direct, data)
                        direct.append(data)
                    elif segment == 'Email':
                        # email = relative_append(email, data)
                        email.append(data)
                    else:
                        print(row)
                        print(segment)
                        raise ValueError("Couldn't recognise the referral type")
                    total += data
                line_count += 1
        date_index = 0
        dates = []
        for data in referral:
            dates.append(date_index)
            date_index += 1
        # plt.plot(dates, totals, f'{colours[file_count]}--')
        # plt.plot(dates, referral, f'{colours[0]}--', dates, organic, f'{colours[1]}--', dates, direct, f'{colours[2]}--', dates, email, f'{colours[3]}--')
        plt.plot(dates, referral, f'{colours[file_count]}--', dates, organic, f'{colours[file_count]}s', dates, direct, f'{colours[file_count]}^', dates, email, f'{colours[file_count]}+')
        file_count += 1

plt.ylabel('Pageviews')
plt.show()