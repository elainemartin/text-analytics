from nltk.tokenize import sent_tokenize
from nltk.tag.stanford import NERTagger
from nltk.parse.stanford import StanfordParser
from corenlp import StanfordCoreNLP

wsj = open('wsj_0063.txt')

#extract named entities
nerTagger=NERTagger('stanford-ner-2014-08-27/classifiers/english.all.3class.distsim.crf.ser.gz', 'stanford-ner-2014-08-27/stanford-ner.jar')
ner = []
for line in wsj:
	ner.append(nerTagger.tag(unicode(line,errors='ignore').split()))

#parse sentences
paragraph = ""
for line in wsj:
	paragraph += line.replace('\n',' ')
sentences = sent_tokenize(paragraph)
parser = StanfordParser('stanford-parser-full-2014-10-31/stanford-parser.jar','stanford-parser-full-2014-10-31/stanford-parser-3.5.0-models.jar')
parsed = parser.raw_parse_sents(sentences)

#coreference
corenlp_dir = "stanford-corenlp-full-2014-08-27"
corenlp = StanfordCoreNLP(corenlp_dir)
corenlp.batch_parse(paragraph)

wsj.close()