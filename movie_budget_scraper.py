import csv
from bs4 import BeautifulSoup
import requests

urls = ['https://www.the-numbers.com/movie/budgets/all/{}'.format(x) for x in range(1, 5802, 100)]

for url in urls:
    page = requests.get(url)
    html = page.text

    outfile = open("movie_budgets.csv","a",newline='')
    writer = csv.writer(outfile)

    tree = BeautifulSoup(html,"lxml")
    table_tag = tree.select("table")[0]
    tab_data = [[item.text for item in row_data.select("th,td")]
                    for row_data in table_tag.select("tr")]

    for data in tab_data:
        writer.writerow(data)
        print(' '.join(data))