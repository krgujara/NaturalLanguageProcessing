import nltk
import re
from nltk.tokenize import sent_tokenize



from nltk.corpus import PlaintextCorpusReader
mycorpus = PlaintextCorpusReader('.', '.*\.txt')
type(mycorpus.fileids())



num_docs = len(mycorpus.fileids())  # mycorpus.fileids() gives the list of all the files
num_docs

# extract the contents of all the files in the file_contents array
file_ids = mycorpus.fileids()
file_contents = []  # file_contents is the array of all the file contents
for i in range(len(file_ids)):
    file = file_ids[i]
    file_handle = open(file,'r',encoding = "ISO-8859-1")
    file_content = file_handle.read()
    # file content has the actual content of each of the file
    file_contents.append(file_content)
    file_handle.close()

print(len(file_contents))
#print(file_contents[5])



# grants_amount is a list of all the grants
grant_amounts = []
# this information is used to get the organization granting the max grant
organization_giving_max_grant = ''


# code for Question 2

resultfile = open('result1.txt', 'w+')
for i in range(0,len(file_contents)-1):
    shorttext = file_contents[i]

    # extract the file text
    pword = re.compile('File *: *(\w+)')
    file_field = re.findall(pword, shorttext)
    resultfile.write(file_field[0])
    
    resultfile.write('\t')
    pword = re.compile('NSF Org *: *(\w+)')
    nsf_field = re.findall(pword, shorttext)
    resultfile.write(nsf_field[0])
    
    resultfile.write(' ')
    pword = re.compile('Total Amt\. *: *([$]\d+)')
    amount_field = re.findall(pword, shorttext)
    resultfile.write(amount_field[0])
    
    
    # getting the amount in $ to find out the maximum grant amount 
    # from all the awards (I used this information to describe the text)
    # in question 2A.
    pword = re.compile('Total Amt\. *: *[$](\d+)')
    res = re. findall(pword, shorttext)
    grant_amounts.append(int(res[0]))
    if (int(res[0]) == 18806079):
        organization_giving_max_grant = nsf_field[0]
    
    resultfile.write(' ')
    pword = re.compile('Abstract *: *\n[ \t]*((?s).*)')
    # text contains string with extra white spaces and \n characters
    text = re.findall(pword, shorttext)
    newtext = text[0].replace('\n','')
    abstract_field = ' '.join(newtext.split())
    resultfile.write(abstract_field)
    resultfile.write('\n')
resultfile.close()



# code for Question 3

# min_abstract and max_abstract len is used to find the max len of the abstract 
# in the given data set
min_abstract_len = 99999
max_abstract_len = 0


resultfile = open('result2.txt', 'w+')
resultfile.write('Abstract_ID | Sentence_No | Sentence\n')
resultfile.write('--------------------------------------\n')
for i in range(0,len(file_contents)-1):
    shorttext = file_contents[i]

    # extract the file text
    pword = re.compile('File *: *(\w+)')
    file_field = re.findall(pword, shorttext)    
    
    pword = re.compile('Abstract *: *\n[ \t]*((?s).*)')
    # text contains string with extra white spaces and \n characters
    text = re.findall(pword, shorttext)
    newtext = text[0].replace('\n','')
    abstract_field = ' '.join(newtext.split())

    sent_tokenize_list = sent_tokenize(abstract_field)

    for i in range (len(sent_tokenize_list)):
        resultfile.write(file_field[0])
        resultfile.write('|')
        resultfile.write(str(i+1))
        resultfile.write('|')
        resultfile.write(sent_tokenize_list[i])
        resultfile.write('\n')
    last_line = 'Number of sentences : ' + str(len(sent_tokenize_list)) + '\n'    
    max_abstract_len = max(max_abstract_len,len(sent_tokenize_list))
    min_abstract_len = min(min_abstract_len, len(sent_tokenize_list))
    resultfile.write(last_line)

resultfile.close()




# code to analyze the given dataset

print('max grant amount : '+ str(max(grant_amounts)))
print('min grant amount : '+ str(min(grant_amounts)))
print('organizarion giving max grant : ' + organization_giving_max_grant)
print('max abstract length : ' + str(max_abstract_len))
print('min abstract length : ' + str(min_abstract_len))
