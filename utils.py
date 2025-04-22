import requests
import csv
from bs4 import BeautifulSoup as bs

#url = "https://en.wiktionary.org/api/rest_v1/page/definition/ghost"
#response = requests.get(url)
#
#print(response.status_code)
#
#word_data = response.json()
#
#for i in word_data['en']:
#    for j in i['definitions']:
#        print(j.keys())

def get_defns(word):
    url = "https://en.wiktionary.org/api/rest_v1/page/definition/"+word
    response = requests.get(url)
    if response.status_code != 200:
       raise Exception("Wiki not found! "+word) 
    word_data = response.json()
    defn_text = []
    for i in word_data['en']:
        for j in i['definitions']:
            soup = bs(j['definition'], 'html.parser')
            defn_text.append(soup.get_text().lower())
    return defn_text

#def get_quotes(word):
#    url = "https://en.wiktionary.org/wiki/"+word
#    response = requests.get(url)
#    soup = bs(response.content, 'html.parser')
#    quotes = soup.find_all('span', {'class':'Latn e-quotation cited-passage'})
#
#    quote_text = ["Word: "+word]
##    word_pos = []
#    for i in quotes:
#        quote_text.append(i.get_text().lower())
##    for q in quote_text:
##        word_pos.append(q.find(word))
#
#    return quote_text

def read_csv(file_path):
    data_array = []
    try:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data_array.append(row)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return []
    except Exception as e:
         print(f"An error occurred: {e}")
         return []
    return data_array
