import csv

#i used this to reformat the csv map, not important
def main():
	w = csv.writer(open("output1.csv", "w"))
	with open('output.csv') as output:
		reader = csv.reader(output)
		for row in reader:
			budget_title = row[0]
			title = row[1]
			title = title.replace(':', '/')
			w.writerow([budget_title, title])

if __name__ == '__main__':
	main()

