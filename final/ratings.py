import requests
import re
import pandas as pd
from bs4 import BeautifulSoup

ratings = []
share = []
viewers = []

for i in range(1,10):
	url = 'http://newmusicandmore.tripod.com/friendsratings0'+str(i)+'.html'
	at = requests.get(url)
	soup = BeautifulSoup(at.text)

	table = soup.find('table')
	rows = table.find_all('tr')
	del(rows[len(rows)-1]) #delete summary row
	del(rows[0]) #delete header

	for row in rows:
		columns = row.find_all('td')
		ratingsshare = columns[4].get_text().strip()
		ratings.append(ratingsshare[0:ratingsshare.find("/")])
		share.append(ratingsshare[ratingsshare.find("/")+1:])
		viewers.append(columns[5].get_text().strip())

url = 'http://newmusicandmore.tripod.com/friendsratings10.html'
at = requests.get(url)
soup = BeautifulSoup(at.text)

table = soup.find('table')
rows = table.find_all('tr')
del(rows[len(rows)-1])
del(rows[0]) #delete header

for row in rows:
	columns = row.find_all('td')
	ratingsshare = columns[4].get_text().strip()
	ratings.append(ratingsshare[0:ratingsshare.find("/")])
	share.append(ratingsshare[ratingsshare.find("/")+1:])
	viewers.append(columns[5].get_text().strip())

output = pd.DataFrame(ratings,columns=['ratings'])
output['share'] = share
output['viewers'] = viewers
output.to_csv("ratings.csv")

