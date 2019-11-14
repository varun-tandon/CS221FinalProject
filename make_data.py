import csv
import pdfkit

def main():
    dic = dict()
    w = csv.writer(open("data.csv", "w"))
    w.writerow(["MovieTitle", "ScriptLink","ProductionBudget", "DomesticGross", "WorldwideGross", "Profit"])
    #pdfkit.from_file('avatar.html', 'avatar.pdf')

    with open('movie_budgets.csv') as budgets:
        with open('output.csv') as scripts:
            script_reader = csv.reader(scripts)
            script_finder = csv.reader(budgets)
            for row in script_reader:
                title = row[0].lower()
                script = row[1].lower()
                for row in script_finder:
                    if (row[2].lower() == title):
                        production_budget = row[3].replace('$', '')
                        domestic_gross = row[4].replace('$', '')
                        worldwide_gross = row[5].replace('$', '')

                        #prod_float = float(production_budget)
                        #print(prod_float)

                        w.writerow([title, title.replace(' ', '-') + '.html',production_budget, domestic_gross, worldwide_gross])
                        #w.writerow([title, title.replace(' ', '-') + '.html',production_budget, domestic_gross, worldwide_gross, prod_float])
                        break

if __name__ == '__main__':
    main()
