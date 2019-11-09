import csv
from bs4 import BeautifulSoup
import requests
from os import path
import random
import time
import sys

def retrieve_movie_script_links():
    page = requests.get('https://www.imsdb.com/all%20scripts/')
    html = page.text

    tree = BeautifulSoup(html,"lxml")
    table_tag = tree.select("table")[1]
    script_column = table_tag.select("td")[81]
    movies = script_column.select("p")
    script_links = {} # map movies to hrefs
    for movie in movies:
        a_data = movie.select("a")[0]
        title = a_data['title']
        title = title[:title.find('Script') - 1]
        script_links[title] = convert_script_link(a_data['href'])

    return script_links

def convert_script_link(url):
    url = url.replace('Movie Scripts', 'scripts')
    url = url.replace(' ', '-')
    url = url[:url.find('-Script')] + url[url.find('.html'):]
    return url

def retrieve_and_store_script(script_url, output_file_name):
    page = requests.get("https://www.imsdb.com" + script_url)
    print("https://www.imsdb.com" + script_url)
    
    html = page.text
    tree = BeautifulSoup(html, 'lxml')
    pre_tags = tree.select('pre')

    if path.exists(output_file_name):
        print(output_file_name, "exists")
        output_file_name += str(random.randint(0, 100))
    
    with open(output_file_name, 'w') as f:        
        f.write(str(pre_tags[0]))

script_links = retrieve_movie_script_links()

i = 0
for movie_title, script_link in script_links.items():
    formatted_movie_title = movie_title.lower()
    formatted_movie_title = formatted_movie_title.replace(" ", "-")
    try:
        retrieve_and_store_script(script_link, "movie_scripts/{}.html".format(formatted_movie_title))
    except IndexError:
        print("Malformed page: {}".format(script_link))
        with open('failed_downloads.txt', 'a') as f:
            f.write('{}: {}'.format(movie_title, script_link))
    except:
        print("Unknown other error", sys.exc_info()[0])
    i += 1
    if (i % 4 == 0):
        time.sleep(1)