import requests
import os.path
import pandas as pd
import re
from bs4 import BeautifulSoup
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.reading import IndexReader
from numpy import sum, log2
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from corenlp import StanfordCoreNLP,batch_parse
# from corenlp_sentiment import StanfordNLPSentimentClient
from scipy.misc import imread
# import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

#get scripts
mainurl = 'http://www.livesinabox.com/friends/scripts.shtml'
at = requests.get(mainurl)
soup = BeautifulSoup(at.text)

seasons = soup.find_all('ul')
urls = []
seasonNum = []
titles = []
for season in seasons:
	episodes = season.find_all('li')
	for episode in episodes:
		urls.append(episode.find('a').get('href'))
		titles.append(episode.find('a').get_text())
		seasonNum.append(seasons.index(season)+1)

episodeNum = []
for title in titles:
	episodeNum.append(title[8:title.find(":")].strip())

speakers = []
text = []
episode = []
for url in urls:
	at = requests.get('http://www.livesinabox.com/friends/'+url)
	soup = BeautifulSoup(at.text)
	lines = soup.find_all('p')
	for line in lines:
		l = line.get_text()
		speaker = re.search(':',l)
		if speaker is not None and l[0]!='{' and l[0]!='(' and l[0]!='[':
			speakers.append(l[0:l.find(":")])
			text.append(l[l.find(":")+2:])
			episode.append(episodeNum[urls.index(url)])

#clean scripts
for i,s in enumerate(speakers):
	notClean = re.search('\(',s)
	if s == 'RACH':
		speakers[i] = unicode('rachel')
	elif s == 'CHAN':
		speakers[i] = unicode('chandler')
	elif s == 'MNCA':
		speakers[i] = unicode('monica')
	elif s == 'PHOE':
		speakers[i] = unicode('phoebe')
	else:
		if notClean is not None:
			speakers[i] = s[0:s.find("(")].lower().strip()
		else: 
			speakers[i] = s.lower().strip()
# cleanSpeakers = [s[0:s.find("(")].lower().strip() for s in speakers]
speakersdf = pd.DataFrame(speakers)
uniqueSpeakers = list(pd.unique(speakersdf.values.ravel()))

#index
schema = Schema(speaker=ID(stored=True),episode=ID(stored=True),line=TEXT(stored=True))
if not os.path.exists("index"):
	os.mkdir("index")
ix = create_in("index", schema)
writer = ix.writer()

for s,e,l in zip(speakers,episode,text):
	writer.add_document(speaker=s,episode=e,line=l)
writer.commit()

#group lines by speaker
searcher = ix.searcher()
lines = []
for s in uniqueSpeakers:
	paragraph = ""
	for d in searcher.documents(speaker=s):
		paragraph += re.sub(r'\([^)]*\)', '',d['line'])+' '
	lines.append(paragraph.replace('\n','').replace('                           ',' '))

schema2 = Schema(speaker=ID(stored=True),line=TEXT(stored=True))
if not os.path.exists("speakerindex"):
	os.mkdir("speakerindex")
ix2 = create_in("speakerindex", schema2)
writer = ix2.writer()

for s,l in zip(uniqueSpeakers,lines):
	writer.add_document(speaker=s,line=l)
writer.commit()

ir2 = ix2.reader()
longFormat = []
for i in ir2.iter_postings():
	if i[0]=='line':
		longFormat.append(i[1:4])
df = pd.DataFrame(longFormat)
df.columns = ['term','doc','count']
T = df.pivot(index='term',columns='doc',values='count')
T = T.fillna(0)

nameIndex = list(T.columns)
names = []
for i in nameIndex:
	names.append(uniqueSpeakers[i])
T.columns = names

mainChars = ['monica', 'joey', 'chandler', 'phoebe', 'ross', 'rachel']
Tmain = pd.DataFrame(T,columns=mainChars)

#naive-bayes
mi = pd.DataFrame()
for c in mainChars:
	N00count = lambda row: sum([row[j] for j in range(0,len(mainChars)-1) if row[j]==1])
	N10count = lambda row: sum([row[j] for j in range(0,len(mainChars)-1) if row[j]!=1])
	N10 = Tmain.drop(c,axis=1).apply(N10count,axis=1)+1
	N00 = sum(Tmain.drop(c,axis=1).apply(N00count,axis=1),axis=0)+1
	N11 = Tmain[c]+1
	N01 = sum([Tmain[c].iloc[j] for j in range(0,len(Tmain)) if Tmain[c].iloc[j]==1],axis=0)+1
	N = N01+N10+N11+N00
	N1dot = N10+N11
	N0dot = N00+N01
	Ndot1 = N01+N11
	Ndot0 = N00+N10
	mi[c] = N11/N*log2(N*N11/(N1dot*Ndot1))+N01/N*log2(N*N01/(N0dot*Ndot1))+N10/N*log2(N*N10/(N1dot*Ndot0))+N00/N*log2(N*N00/(N0dot*Ndot0))

uniqueWords = pd.DataFrame()
for c in mainChars:
	mi = mi.sort(c,ascending=False)
	uniqueWords[c] = mi.index[0:30]

uniqueTerms = pd.unique(uniqueWords.values.ravel())
nonUnique = [item for item, count in Counter(uniqueWords.values.ravel()).iteritems() if count > 1]
keep = set(uniqueTerms)-set(nonUnique)

featureSelection = pd.DataFrame(Tmain,index=keep)
train = pd.DataFrame(Tmain,index=keep).transpose()
train.to_csv("uniqueWords.csv")
mnb = MultinomialNB()
mnbFit = mnb.fit(train,mainChars)

# ix = open_dir("index")
ir = ix.reader()
longFormat2 = []
for i in ir.iter_postings():
	if i[0]=='line':
		longFormat2.append(i[1:4])
df2 = pd.DataFrame(longFormat2)
df2.columns = ['term','doc','count']
Tlines = df2.pivot(index='term',columns='doc',values='count')
# Tlines = Tlines.fillna(0)
testIndex = list(Tlines.columns)
testanswers = []
for i in testIndex:
	testanswers.append(speakers[i])
Tlines.columns = testanswers
# maintest = filter(lambda x: x in mainChars,testanswers)

testlines = pd.DataFrame(Tlines,index=keep).transpose()
testlines = testlines.loc[mainChars]
testlines = testlines.fillna(0)
pred = mnb.predict(testlines)
mnb.score(testlines,testlines.index) #0.20776382878443805
cm = pd.DataFrame(confusion_matrix(testlines.index,pred))
cm.to_csv("confusion.csv")
#term frequency
#remove words

#group lines by episode
episodeLines = []
for e in episodeNum:
	# paragraph = inde	# for d in searcher.documents(episode=e):
	# 	paragraph += d['speaker']+': '+d['line']+' '
	# 	# paragraph += re.sub(r'\([^)]*\)', '',d['line'])+' '
	# episodeLines.append(paragraph.replace('\n','').replace('                           ',' '))
	outfile = open('scripts/'+e+'.txt','w')
	paragraph = ""
	for d in searcher.documents(episode=e):
		outfile.writelines(d['line'].encode('utf-8')+' ')
		# outfile.writelines((d['speaker']+': '+d['line']).encode('utf-8')+' ')
	# 	paragraph += d['speaker']+': '+d['line']+' '
	# 	# paragraph += re.sub(r'\([^)]*\)', '',d['line'])+' '
	# paragraph = paragraph.replace('\n','').replace('                           ',' ')
	# outfile.writelines(paragraph.encode('utf-8'))
	outfile.close()

parsed = []
corenlp_dir = "stanford-corenlp-full-2014-08-27"
corenlp = StanfordCoreNLP(corenlp_dir)
for e in episodeNum:
	for d in searcher.documents(episode=e):
		parsed.append(corenlp.raw_parse(d))

# sentClient = StanfordNLPSentimentClient('http://localhost:8080')
# sentiment = []
# for t in text:
# 	sentiment.append(sentClient.classify(t))

# mask = imread("friends.gif")
wc = WordCloud(max_words=30,stopwords=STOPWORDS|{'s','t','m','re','oh','right','don','know','well','hey','gonna','okay','yeah','go','really','think','hi','uh','look','god','mean','one','ye','guy','y','got','come','now'},font_path='/Users/elaine/Library/Fonts/Berlin.ttf')
for c in mainChars:
	wc.generate(lines[uniqueSpeakers.index(c)])
	wc.to_file(c+".png")

# wc = WordCloud(background_color="white",max_words=50,mask=mask,stopwords=STOPWORDS|{'s','t','m','re','oh','right','don','know','well','hey','gonna','okay','yeah','go','really','think','hi','uh','look','god','mean','one','ye','guy','y','got','come','now'},font_path='/Users/elaine/Library/Fonts/Berlin.ttf')
# for c in mainChars:
# 	wc.generate(lines[uniqueSpeakers.index(c)])
# 	wc.to_file(c+".png")

#group lines by speaker & episode
searcher = ix.searcher()
mainChars = ['monica', 'joey', 'chandler', 'phoebe', 'ross', 'rachel', 'mike', 'janice', 'gunther']
episodeChar = []
for c in mainChars:
	if not os.path.exists(c):
		os.mkdir(c)
	for e in episodeNum:
		outfile = open(c+'/'+e+'.txt','w')
		for d in searcher.documents(speaker=c,episode=e):
			outfile.writelines(d['line'].encode('utf-8')+' ')
		outfile.close()


# schema3 = Schema(episode=ID(stored=True),line=TEXT(stored=True))
# if not os.path.exists("episodeindex"):
# 	os.mkdir("episodeindex")
# ix3 = create_in("episodeindex", schema3)
# writer = ix3.writer()

# for s,l in zip(uniqueSpeakers,episodeLines):
# 	writer.add_document(speaker=s,line=l)
# writer.commit()

# ir2 = ix2.reader()
# longFormat = []
# for i in ir2.iter_postings():
# 	if i[0]=='line':
# 		longFormat.append(i[1:4])
# df = pd.DataFrame(longFormat)
# df.columns = ['term','doc','count']
# T = df.pivot(index='term',columns='doc',values='count')
# T = T.fillna(0)

#normalize for number of episodes in each season
#who has most lines/mentions per season
#coreference
#classify season 10 lines based on parsing
#length of sentences/vocabulary
#sentiment over time
#overlay relationships
#central perk vs monica's apt vs joey's apt
#who is talking to whom/about
#if speaker is in title, does speaker have more lines?
#thanksgiving episodes
#special guests