import re

bios = open('classbios.txt')
timepoints = open('classbios_timepoints.txt','w')

pattern = '\d\d\d\d|today |Today | day |(Y|y)esterday | now|Jan|Feb|Apr|Jun|Jul|August|Sep|Oct|Nov|Dec|Monday|Tue|Wed|Thurs|Friday|Saturday|Sunday| years* ^(old)| last | Last |(((T|t)his|(N|n)ext) (winter|spring |summer))' #remove everyda

wholetext = ""

for line in bios:
	dashes = re.search('-----', line)
	if dashes is None:
		wholetext += line.replace('\n',' ')

sentences = re.split('[.?!]', wholetext)

for sentence in sentences:
	match = re.search(pattern, sentence)
	if match is not None:
		timepoints.writelines(sentence+'\n')

bios.close()
timepoints.close()