import os.path
import pandas as pd
from re import search, sub
from whoosh.index import create_in#, open_dir
from whoosh.fields import *
from whoosh.reading import IndexReader
from numpy import log10, square, sqrt
from nltk.tag.stanford import POSTagger

bios = open('classbios.txt')
output = open('classbios_summaries.txt','w')

#split into paragraphs
people = []
paragraph = ""

for line in bios:
	dashes = search('-----', line)
	if dashes is None:
		paragraph += line.replace('\n',' ')
	else:
		people.append(paragraph)
		paragraph = ""
else:   
	people.append(paragraph)

people = filter(lambda a: a != "", people)

bios.close()

#whoosh
#index
schema = Schema(number=NUMERIC(stored=True),content=TEXT(stored=True))
if not os.path.exists("index"):
	os.mkdir("index")
ix = create_in("index", schema)
writer = ix.writer()

for person in people:
	writer.add_document(number=people.index(person)+1,content=unicode(person,errors='ignore'))
writer.commit()

ir = ix.reader()

#produce term-document matrix (T)
longFormat = []
for i in ir.iter_postings():
	if i[0]=='content':
		longFormat.append(i[1:4])
df = pd.DataFrame(longFormat)
df.columns = ['term','doc','count']
T = df.pivot(index='term',columns='doc',values='count')
T = T.fillna(0)
unsortedT = T
T['max'] = T.max(axis=1)
T = T.sort('max',ascending=False)
T.to_csv('T.csv')
T['total'] = T.sum(axis=1)
T= T.sort('total',ascending=False)

#output top 10 terms
# ir.most_frequent_terms("content",number=10)

#produce normalized matrix (M)
N = 42
countDocs = lambda row: sum([int(row[i]!=0) for i in range(0,N)])
unsortedT['df'] = unsortedT.apply(countDocs,axis=1)
M = 1+log10(unsortedT.iloc[:,0:N])
M = M.replace('-inf',0).fillna(0)
for i in range(0,N):
	M[i] = M[i]*(log10(N/unsortedT['df']))
Msquared = M.applymap(lambda x: square(x))
Msquared = Msquared.append(sqrt(Msquared.sum(axis=0)),ignore_index=True)
for i in range(0,2535):
	M.iloc[i] = M.iloc[i]/Msquared.iloc[2535]
M['max'] = M.max(axis=1)
M= M.sort('max',ascending=False)
M.to_csv('M.csv')
M['total'] = M.sum(axis=1)
M= M.sort('total',ascending=False)

#output top 10 terms
# ir.most_distinctive_terms("content",number=10)

#summarize documents
k = 5
Tsummaries = []
Msummaries = []
for i in range(0,N):
	T = T.sort(i,ascending=False)
	Tsummaries.append(list(T.index[0:k]))
for i in range(0,N):
	M = M.sort(i,ascending=False)
	Msummaries.append(list(M.index[0:k]))

#output examples
for summary in Tsummaries:
	output.writelines('Document '+str(Tsummaries.index(summary))+': '+str(summary)+'\n')
output.close()

#extract parts of speech
posTagger =POSTagger('stanford-postagger-2014-08-27/models/english-bidirectional-distsim.tagger','stanford-postagger-2014-08-27/stanford-postagger.jar')
pos = []
for person in people:
	pos.append(posTagger.tag(unicode(person,errors='ignore').split()))
