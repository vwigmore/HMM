from glob import glob
import fileinput
import json
import ast
from seqlearn.datasets import load_conll

#f = open("E:\TUDelft\FirstQuarter\AI\project2\conceder_conceder.json","r")
#print(f.read())

def features(sentence, i):
    """Features for i'th token in sentence.
    Currently baseline named-entity recognition features, but these can
    easily be changed to do POS tagging or chunking.
    """

    word = sentence[i]

    yield "word:{}" + word.lower()

    if word[0].isupper():
        yield "CAP"

    if i > 0:
        yield "word-1:{}" + sentence[i - 1].lower()
        if i > 1:
            yield "word-2:{}" + sentence[i - 2].lower()
    if i + 1 < len(sentence):
        yield "word+1:{}" + sentence[i + 1].lower()
        if i + 2 < len(sentence):
            yield "word+2:{}" + sentence[i + 2].lower()

files = open("E:\TUDelft\FirstQuarter\AI\project2\conceder_conceder.json","r")
f2 = files.read()
loaded_json = json.loads(f2)
#a = ast.literal_eval(loaded_json)
#print(type(loaded_json))
#print(loaded_json)
#for key in loaded_json.keys():
issues_str = loaded_json['issues']
Utility2_str = loaded_json['Utility2']
Utility1_str = loaded_json['Utility1']
bids_str = loaded_json['bids']
print(issues_str)
for key in issues_str.keys():
    Fruit_str = issues_str[key]
    print(key)
    print(Fruit_str)

for key2 in Utility2_str.keys():
    b_str = Utility2_str[key2]
    print(key2)
    print(b_str)

for key3 in Utility1_str.keys():
    c_str = Utility1_str[key3]
    print(key3)
    print(c_str)

for key4 in bids_str:
    d_str = key4
    print(d_str)
    #print(d_str)
#for key4 in bids_str.keys():
#    d_str = bids_str[key4]
#    print(key4)
#    print(d_str)
