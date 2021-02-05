from __future__ import division, print_function
import numpy
import json
from collections import Counter
import pymongo
import operator

def create_dataset(db):
    # load training set
    train = json.load(open('train.json'))
    train_set = []
    ingredients = []
    cuisines = Counter([i['cuisine'] for i in train])
    cuisine_ingreds_count = dict.fromkeys(cuisines.keys(), [])

    for i in train:
        if ''.join(i["ingredients"]).__contains__('.'):
            pass
        else:
            train_set.append(i)
            ingredients.extend(i["ingredients"])
            if not cuisine_ingreds_count[i["cuisine"]]:
                cuisine_ingreds_count[i["cuisine"]] = i['ingredients']
            else:
                cuisine_ingreds_count[i['cuisine']].extend(i['ingredients'])

    # record recipes in each cuisine
    cuisine_total = {i:len(cuisine_ingreds_count[i]) for i in cuisine_ingreds_count.keys() }
    train_total = len(train_set)

    for i in cuisine_ingreds_count.keys():
        cuisine_ingreds_count[i] = Counter(cuisine_ingreds_count[i])

    # total of each ingredient in training set
    ingred_tot_count = Counter(ingredients)

    # total of each ingredient in each cuisine in training set
    ingred_cuisine_count = dict.fromkeys(ingred_tot_count.keys(), dict.fromkeys(cuisine_ingreds_count.keys(), int))

    for i in ingred_tot_count.keys():
        ingred_cuisine_count[i] ={j: cuisine_ingreds_count[j][i] for j in cuisine_ingreds_count.keys() if i in cuisine_ingreds_count[j].keys()}

    priors = {i:cuisines[i]/train_total for i in cuisine_ingreds_count.keys()}

    db.create_collection('ingred_info')
    db.create_collection('cuisine_info')
    db.create_collection('training_set_info')

    map(lambda x: db.ingred_info.insert_one({'ingredient': x, 'cuisines': ingred_cuisine_count[x], 'total': ingred_tot_count[x] }), ingred_cuisine_count.keys())
    map(lambda x: db.cuisine_info.insert_one({'cuisine': x, 'ingredients': cuisine_ingreds_count[x], 'count': cuisines[x], 'total_ingred': cuisine_total[x],'prior': priors[x] }), cuisine_ingreds_count.keys())

    db.training_set_info.insert_one({'total_recipes': train_total, 'ingred_vocab': len(ingred_cuisine_count.keys())})


def predictor_score(ingredients, db):
    #identify new ingredients
    ingred = {i['ingredient']:i for i in db.ingred_info.find()}
    new_ingred = set(ingredients).difference(ingred.keys())
    result = ''
    exist_ingred = set(ingredients).difference(new_ingred)
    ingred_info = {i: db.ingred_info.find_one({'ingredient': i}) for i in list(exist_ingred)}
    cuisine_info = {i['cuisine']: i for i in db.cuisine_info.find()}
    liklihood = dict.fromkeys(cuisine_info.keys(),[])
    posterior_cuisine = dict.fromkeys(cuisine_info.keys())
    ingred_vocab =  db.training_set_info.find_one()['ingred_vocab']

    smoothing = 1

    # creaate liklihood first with new ingredients
    for i in cuisine_info.keys():
        liklihood[i] = [(smoothing/(cuisine_info[i]['total_ingred'] + ingred_vocab)) ** len(new_ingred)]
    # append liklihoods for existing ingredients
    for i in cuisine_info.keys():
        for j in ingred_info.keys():
            if i in ingred_info[j]['cuisines'].keys():
                liklihood[i].append((ingred_info[j]['cuisines'][i] + smoothing)/(cuisine_info[i]['total_ingred'] + ingred_vocab))
            else:
                liklihood[i].append(smoothing/(cuisine_info[i]['total_ingred'] + ingred_vocab))
    total = 0;
        # posterior probability
    for i in posterior_cuisine.keys():
        posterior_cuisine[i] = cuisine_info[i]['prior'] * numpy.prod(liklihood[i])
        total += posterior_cuisine[i];

    posterior = [{"cuisine": a, "posterior": posterior_cuisine[a]/total * 100} for a in posterior_cuisine.keys()]

    result = [{'predicted': max(posterior_cuisine.iteritems(), key = operator.itemgetter(1))[0]}]
    result.extend(posterior)
    if len(new_ingred) == len(ingredients):
        return ''
    else:
        return result

if __name__ == "__main__":
    # Connect To Db
    try:
        conn = pymongo.MongoClient('localhost', 27017)
        print("Connected")
    except pymongo.errors.ConnectionFailure, e:
        print(u"could not connect to db {}").format(e)
    db = conn['cuisine']
    #create_dataset(db)

    print(predictor_score([  "flour",
      "curds",
      "tumeric",
      "chili powder",
      "mustard",
      "seeds",
      "oil",
      "water",
      "salt"], db))
