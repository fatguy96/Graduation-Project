import csv

csv_reader = csv.reader(open('data/data_org.csv'))

i = 0
for row in csv_reader:
    i += 1

print(i)