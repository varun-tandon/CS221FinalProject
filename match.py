from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import csv
from collections import defaultdict

def main():
	#options = ['I, Robot', 'Robotics', 'Robots', 'housewives']
	#highest = process.extract('i,robot', options)
	#print(highest)

	#options.remove(highest[0][0])
	#print(options)
	dic = dict()
	file_titles = list()

	with open('file_titles.txt') as titles:
		content = titles.readlines()
		print(content)

		file_titles = [x[:-6].replace('-', ' ') for x in content]

	total = 0
	missing = 0
	duplicates = defaultdict(int)

	w = csv.writer(open("output.csv", "w"))

	start = True
	with open('movie_budgets.csv') as budgets:
		reader = csv.reader(budgets)
		for row in reader:
			if start:
				start = not start
				continue
			if total > 200:
				break

			budget_title = row[2]
			tup = process.extractOne(budget_title, file_titles)
			title, score = tup

			if score >= 90:
				#if it's a version of a film, has to be exact
				nums = '0123456789'
				last_char1 = budget_title[-1]
				last_char2 = title[-1]

				if (last_char1 in nums) != (last_char2 in nums):
					continue

				if (':' in budget_title) != (':' in title):
					continue

				######
				dic[budget_title] = title
				duplicates[title] += 1
				w.writerow([budget_title, title.replace(' ', '-') + '.html'])
			else:
				missing += 1
				print('missing area')
				print(budget_title)
				print(title)
				print(score)
				print()

			total += 1

	print('total:', total)
	print('missing:', missing)

	print('duplicates')
	for d in duplicates:
		if duplicates[d] > 1:
			print(f'key {d}, value {duplicates[d]}')

	print('dict')
	print(dic)


if __name__ == '__main__':
	main()