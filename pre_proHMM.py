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

def read_round(sentence):
    for x in sentence:
        print(x)

def get_agent1(sentence):
    #util of fruit
    print(sentence[1])

    #util of juice


    #util of topping1

    #util of topping2
    #print(sentence)

def get_fruit_weight(sentence):
    weight_val = 0;
    for key2 in sentence.keys():
        if key2 == 'Fruit':
            fruit_str = sentence[key2]
            for weight in fruit_str.keys():
                if weight == 'weight':
                    weight_val = int(fruit_str[weight])

    return weight_val

def get_juice_weight(sentence):
    for key2 in sentence.keys():
        if key2 == 'Juice':
            juice_str = sentence[key2]
            for weight in juice_str.keys():
                if weight == 'weight':
                    weight_val = juice_str[weight]


                    print(weight_val)

def get_Topping1_weight(sentence):
    for key2 in sentence.keys():
        if key2 == 'Topping1':
            topping1_str = sentence[key2]
            for weight in topping1_str.keys():
                if weight == 'weight':
                    weight_val = topping1_str[weight]


                    print(weight_val)

def get_Topping2_weight(sentence):
    for key2 in sentence.keys():
        if key2 == 'Topping2':
            topping2_str = sentence[key2]
            for weight in topping2_str.keys():
                if weight == 'weight':
                    weight_val = topping2_str[weight]


                    print(weight_val)


def cal_util(sentence,sentence2,fruit_list,juice_list,topping1_list,topping2_list):
    #agent1 utility
    for i in sentence:
        round_util = dict(i)
        if 'agent1' in round_util:
            agent1 = round_util['agent1']
            agent1_list = agent1.split(',')
            fru_util = 0;
            jui_util = 0;
            top1_util = 0;
            top2_util = 0;
            for val in agent1_list:
                if val in fruit_list:
                    fru_util = get_fruit_weight(sentence2);
                    print('the fruit is',fru_util)
                if val in juice_list:
                    jui_util = get_juice_weight(sentence2);
                if val in topping1_list:
                    top1_util = get_Topping1_weight(sentence2);
                if val in topping2_list:
                    top2_util = get_Topping2_weight(sentence2)

            util = fru_util + jui_util + top1_util + top2_util
            #print("the round of util is:",util);

def main():
    files = open("E:\TUDelft\FirstQuarter\AI\project2\conceder_conceder.json", "r")
    f2 = files.read()
    loaded_json = json.loads(f2)
    # a = ast.literal_eval(loaded_json)
    # print(type(loaded_json))
    # print(loaded_json)
    # for key in loaded_json.keys():
    issues_str = loaded_json['issues']
    utility2_str = loaded_json['Utility2']
    Utility1_str = loaded_json['Utility1']
    bids_str = loaded_json['bids']
    round_list = list()
    fruit_list = list()
    juice_list = list()
    util1_list = list()
    util2_list = list()
    topping1_list = list()
    topping2_list = list()

    for key in issues_str.keys():
        if key == 'Fruit':
            Fruit_str = issues_str[key]
            fruit_list = Fruit_str
        if key == 'Juice':
            Juice_str = issues_str[key]
            juice_list.append(Juice_str)
        if key == 'Topping1':
            topping1_str = issues_str[key]
            topping1_list.append(topping1_str)
        if key == 'Topping2':
            topping2_str = issues_str[key]
            topping2_list.append(topping2_str)

    for key3 in Utility1_str.keys():
        c_str = Utility1_str[key3]
        util1_list.append(key3)
        util1_list.append(c_str)

    for key4 in bids_str:
        d_str = key4
        round_list.append(d_str)

    print("the value is",utility2_str)
    print(get_fruit_weight(utility2_str))
#get_juice_weight(Utility2_str)
#get_Topping1_weight(Utility2_str)
#get_Topping2_weight(Utility2_str)

#cal_util(round_list,Utility2_str);


