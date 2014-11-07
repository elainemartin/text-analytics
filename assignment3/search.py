import os.path
from re import search, sub
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.query import *
from whoosh.qparser import QueryParser

bios = open('classbios.txt')
queryOutput = open('classbios_queryOutput.txt','w')

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
people.remove("")

bios.close()

#create index
schema = Schema(number=NUMERIC(stored=True),content=TEXT(stored=True))
if not os.path.exists("index"):
	os.mkdir("index")
ix = create_in("index", schema)
writer = ix.writer()

for person in people:
		writer.add_document(number=people.index(person)+1,content=unicode(person,errors='ignore'))
writer.commit()

#search
with ix.searcher() as searcher:
	#queries
	queries = []
	for i in range(0,3):
		userInput = input('Enter your query: ')
		queries.append(userInput) 
	#parser
	parser = QueryParser("content",ix.schema)
	for query in queries:
		myquery = parser.parse(query)
		results = searcher.search(myquery)
		queryOutput.writelines('Query: '+query+'\n')
		queryOutput.writelines('Results: '+'\n')
		for result in results:
			queryOutput.writelines(sub('<b class="match term\d">','',result.highlights("content").replace('</b>',''))+'\n')
		queryOutput.writelines('\n')

queryOutput.close()