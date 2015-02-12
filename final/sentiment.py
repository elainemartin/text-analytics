import requests
import pandas as pd
from corenlp import StanfordCoreNLP,batch_parse
from collections import Counter
from bs4 import BeautifulSoup

#parse using corenlp
corenlp_dir = "stanford-corenlp-full-2014-08-27"
parse = batch_parse('scripts',corenlp_dir, raw_output=True)
parsedEpisodes = []
for p in parse:
	try:
		parsedEpisodes.append(p)
	except:
		parsedEpisodes.append('')

del(parsedEpisodes[0:2]) #remove hidden files

#extract sentiment from corenlp results
# allSentiments = []
allValues = []

for e in range(0,len(parsedEpisodes)):
	try:
		sentences = parsedEpisodes[e]['root']['document']['sentences']['sentence']
		sentimentValues = []
		sentiments = []
		for s in sentences:
			try:
				sentimentValues.append(int(s['@sentimentValue'])-2)
				# sentiments.append(s['@sentiment'])
			except:
				# sentimentValues.append('')
				sentimentValues.append('')
		# allSentiments.append(sentiments)
		allValues.append(sentimentValues)
	except:
		# allSentiments.append('')
		allValues.append('')

#get overall sentiment for episode
summedValues = []
# modeSentiment = []
empty = []
wordcount = []

for v in allValues:
	try:
		summedValues.append(sum(v))
		count=len(v)
	except:
		summedValues.append('')
		empty.append(allValues.index(v))
		count=0
	wordcount.append(count)


# for s in allSentiments:
# 	d = Counter(s)
	# modeSentiment.append(d.most_common(1))

#get episode numbers & titles
# mainurl = 'http://www.livesinabox.com/friends/scripts.shtml'
# at = requests.get(mainurl)
# soup = BeautifulSoup(at.text)

# seasons = soup.find_all('ul')
# seasonNum = []
# titles = []
# for season in seasons:
# 	episodes = season.find_all('li')
# 	for episode in episodes:
# 		titles.append(episode.find('a').get_text())
# 		seasonNum.append(seasons.index(season)+1)

# episodeNum = []
# for title in titles:
# 	episodeNum.append(title[8:title.find(":")].strip())

episodes = pd.DataFrame(episodeNum,columns=['episodeNum'])
episodes['titles'] = titles
episodes['seasonNum'] = seasonNum
episodes = episodes.drop(episodes.index[[95,165]])
episodes = episodes.sort('episodeNum')
# episodes['sentiment'] = modeSentiment
episodes['value'] = summedValues
episodes['wordcount']=wordcount
episodes['episodeNum'] = episodes['episodeNum'].astype('int')
episodes = episodes.sort('episodeNum')
# episodes['normSent'] = episodes['value'].div(episodes['wordcount'].astype('int'),axis='index')

episodes.drop('titles',axis=1).to_csv("episodesNeg.csv")

episodes['episodeNum'].to_csv("episodes.csv")
episodes['value'].to_csv("episodesent.csv")
episodes['wordcount'].to_csv("episodewc.csv")

'''
by character
'''
mainChars = ['monica', 'joey', 'chandler', 'phoebe', 'ross', 'rachel', 'mike', 'gunther']
charSent = pd.DataFrame(episodeNum,columns=['episodeNum'])
charSent['titles'] = titles
charSent['seasonNum'] = seasonNum
charSent = charSent.drop(charSent.index[[95,165]])
charSent = charSent.sort('episodeNum')
for c in mainChars:
	parse = batch_parse(c,corenlp_dir, raw_output=True)
	parsedChar = []
	for p in parse:
		try:
			parsedChar.append(p)
		except:
			parsedChar.append('')

	del(parsedChar[0]) #remove hidden file

	#extract sentiment from corenlp results
	# allSentiments = []
	allValues = []

	for e in range(0,len(parsedChar)):
		try:
			sentences = parsedChar[e]['root']['document']['sentences']['sentence']
			sentimentValues = []
			sentiments = []
			for s in sentences:
				try:
					sentimentValues.append(int(s['@sentimentValue'])-2)
					# sentiments.append(s['@sentiment'])
				except:
					sentimentValues.append('')
					# sentiments.append('')
			# allSentiments.append(sentiments)
			allValues.append(sentimentValues)
		except:
			# allSentiments.append('')
			allValues.append('')

	summedValues = []
	# modeSentiment = []
	# empty = []
	wordcount = []

	for v in allValues:
		try:
			summedValues.append(sum(v))
			count=len(v)
		except:
			summedValues.append('')
			# empty.append(allValues.index(v))
			count=0
		wordcount.append(count)


	# for s in allSentiments:
	# 	d = Counter(s)
	# 	modeSentiment.append(d.most_common(1))

	charSent[c+' sent'] = summedValues
	charSent[c+' wordcount']=wordcount

charSent['episodeNum'] = charSent['episodeNum'].astype('int')
charSent = charSent.sort('episodeNum')
# charSent['normSent'] = charSent['value'].div(charSent['wordcount'].astype('int'),axis='index')

charSent.drop('titles',axis=1).to_csv("charSentNeg.csv")





