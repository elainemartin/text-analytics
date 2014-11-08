import requests
import os.path
import pandas as pd
from bs4 import BeautifulSoup
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.query import *
from whoosh.qparser import QueryParser, FuzzyTermPlugin, PhrasePlugin, SequencePlugin
from whoosh.spans import SpanNear
from whoosh.reading import IndexReader
from numpy import log2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

###1
##scrape web
mainurl = 'http://en.wikipedia.org/wiki/List_of_countries_and_capitals_with_currency_and_language'
at = requests.get(mainurl)
soup = BeautifulSoup(at.text)

getContinents = soup.find_all(class_ = 'toctext')
continent = []
for cont in getContinents:
		continent.append(cont.get_text())

tables = soup.find_all(class_ = 'wikitable sortable')
continents=[]
countryURL=[]
capitalURL=[]
countryName = []
capitalName = []
for table in tables:
	rows = table.find_all('tr')
	for row in rows:
		columns = row.find_all('td')
		if len(columns)>3:
			countryURL.append(columns[1].find('a').get('href'))
			try:
				capitalURL.append(columns[2].find('a').get('href'))
				capitalName.append(columns[2].find('a').get_text(strip=True))
				continents.append(continent[tables.index(table)])
				countryName.append(columns[1].find('a').get_text(strip=True))
			except:
				del(countryURL[-1])

countryText = []
for url in countryURL:
	paragraph = ""
	at = requests.get('http://en.wikipedia.org'+url)
	soup = BeautifulSoup(at.text)
	paragraphs = soup.find_all('p')
	for p in paragraphs:
			paragraph += p.get_text()+' '
	countryText.append(paragraph)
capitalText = []
for url in capitalURL:
	paragraph = ""
	at = requests.get('http://en.wikipedia.org'+url)
	soup = BeautifulSoup(at.text)
	paragraphs = soup.find_all('p')
	for p in paragraphs:
			paragraph += p.get_text()+' '
	capitalText.append(paragraph)

##create index
schema = Schema(city_name=ID(stored=True),country_name=ID(stored=True),continent=ID(stored=True),city_text=TEXT(stored=True),country_text=TEXT(stored=True))
if not os.path.exists("indexdir"):
	os.mkdir("indexdir")
ix = create_in("indexdir", schema)
writer = ix.writer()

for i in range(0,len(continents)):
	writer.add_document(city_name=capitalName[i],country_name=countryName[i],continent=continents[i],city_text=capitalText[i],country_text=countryText[i])
writer.commit()

###2
##find cities
with ix.searcher() as searcher:
	parser = QueryParser("city_text",ix.schema)
	#greek & roman -persian
	myquery = parser.parse('Greek AND Roman NOT Persian')
	results = searcher.search(myquery,limit=None)
	for result in results:
		print(result['city_name'])
	#shakespeare incl mispelled
	parser.add_plugin(FuzzyTermPlugin())
	myquery = parser.parse(u'Shakespeare~3')
	results = searcher.search(myquery,limit=None)
	for result in results:
		print(result['city_name'])
	#located below sea level
	# parser.remove_plugin_class(PhrasePlugin)
	# parser.add_plugin(SequencePlugin())
	# myquery = parser.parse("located below sea level~10")
	# myquery = SpanNear.phrase("city_text",["located","below","sea","level"],slop=10)
	myquery = Phrase("city_text",list([unicode("located"),unicode("below"),unicode("sea"),unicode("level")]),slop=10)
	results = searcher.search(myquery,limit=None)
	for result in results:
		print(result['city_name'])

###3
#naive bayes classifier - identify continent using description of city
#extract all terms for all cities from index
ir = ix.reader()
longFormat = []
for i in ir.iter_postings():
	if i[0]=='city_text':
		longFormat.append(i[1:4])
df = pd.DataFrame(longFormat)
df.columns = ['term','doc','count']
T = df.pivot(index='term',columns='doc',values='count')
T = T.fillna(0)

#compute mutual information - use laplacian smoothing
continentsIndex=[]
for i in range(0,len(continents)):
	if continents[i]!=continents[i-1]:
		continentsIndex.append(i)
continentsIndex.append(len(continents))

for i in range(1,len(continentsIndex)):
	countDocs = lambda row: sum([int(row[j]!=0) for j in range(continentsIndex[i-1],continentsIndex[i])])
	T[str(continent[i-1])] = T.apply(countDocs,axis=1)

continent = continent[0:7]
laplaceSmoothing = T.iloc[:,-7:]#.applymap(lambda x: x+1)
mi = pd.DataFrame()
for c in continent:
	N00count = lambda row: sum([row[j] for j in range(0,6) if row[j]==1])
	N10count = lambda row: sum([row[j] for j in range(0,6) if row[j]!=1])
	N10 = laplaceSmoothing.drop(str(c),axis=1).apply(N10count,axis=1)+1
	N00 = sum(laplaceSmoothing.drop(str(c),axis=1).apply(N00count,axis=1),axis=0)+1
	N11 = laplaceSmoothing[str(c)]+1
	N01 = sum([laplaceSmoothing[str(c)].iloc[j] for j in range(0,len(laplaceSmoothing)) if laplaceSmoothing[str(c)].iloc[j]==1],axis=0)+1
	N = N01+N10+N11+N00
	N1dot = N10+N11
	N0dot = N00+N01
	Ndot1 = N01+N11
	Ndot0 = N00+N10
	mi[str(c)] = N11/N*log2(N*N11/(N1dot*Ndot1))+N01/N*log2(N*N01/(N0dot*Ndot1))+N10/N*log2(N*N10/(N1dot*Ndot0))+N00/N*log2(N*N00/(N0dot*Ndot0))

#table w/ 30 most informative terms per continent
mi = mi.fillna(0)
mi.to_csv("mi3.csv")
mostInformative=pd.DataFrame()
for c in continent:
	mi = mi.sort(str(c),ascending=False)
	mostInformative[str(c)]=mi.index[0:30]
mostInformative.to_csv("mostInformative3.csv")

#pick unique terms - how many?
uniqueTerms = pd.unique(mostInformative.values.ravel())
len(uniqueTerms)
#non unique
from collections import Counter
nonUnique = [item for item, count in Counter(mostInformative.values.ravel()).iteritems() if count > 1]
keep = set(uniqueTerms)-set(nonUnique)

#build classifier - choose multinomial or bernoulli
featureSelection = pd.DataFrame(T.iloc[:,-7:],index=keep)
trainCapitals = pd.DataFrame(T.iloc[:,0:-7],index=keep).transpose()
mnb = MultinomialNB()
#classify cities & build confusion matrix
mnbFit = mnb.fit(trainCapitals,continents)
mnb.score(trainCapitals,continents) #0.90347490347490345
predCapitals = mnb.predict(trainCapitals)
cm = pd.DataFrame(confusion_matrix(continents,predCapitals))
cm.to_csv("confusion.csv")

#choose city that's not a capital
SFtext = ''
url = 'https://en.wikipedia.org/wiki/San_Francisco'
at = requests.get(url)
soup = BeautifulSoup(at.text)
paragraphs = soup.find_all('p')
for p in paragraphs:
	SFtext += p.get_text()+' '

schema = Schema(content=TEXT(stored=True))
if not os.path.exists("sfindex"):
	os.mkdir("sfindex")
ix2 = create_in("sfindex", schema)
writer = ix2.writer()
writer.add_document(content=SFtext)
writer.commit()
ir = ix2.reader()
longSF=[]
for i in ir.iter_postings():
	longSF.append(i[1:4])
sfDF = pd.DataFrame(longSF)
sfDF.columns = ['term','doc','count']
sfDF = sfDF.pivot(index='doc',columns='term',values='count')
#feature vector
testSF = pd.DataFrame(sfDF,columns=keep)
testSF = testSF.fillna(0)
testSF.transpose().to_csv("sf.csv")
#classify it
predSF = mnb.predict(testSF)
#probability for each continent
mnb.predict_proba(testSF)