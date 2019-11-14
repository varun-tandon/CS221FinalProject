import csv
from collections import defaultdict

def main():
    dic = dict()
    w = csv.writer(open("data.csv", "w"))
    w.writerow(["MovieTitle", "ScriptLink","ProductionBudget", "DomesticGross", "WorldwideGross"])

    with open('movie_budgets.csv') as budgets:
        with open('output.csv') as scripts:
            script_reader = csv.reader(scripts)
            script_finder = csv.reader(budgets)
            for row in script_reader:
                title = row[0].lower()
                script = row[1].lower()
                for row in script_finder:
                    if (row[2].lower() == title):
                        production_budget = row[3]
                        domestic_gross = row[4]
                        worldwide_gross = row[4]
                        w.writerow([title, title.replace(' ', '-') + '.html',production_budget, domestic_gross, worldwide_gross])
                        break

if __name__ == '__main__':
    main()
