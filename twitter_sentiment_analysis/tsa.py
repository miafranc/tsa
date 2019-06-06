from twython import Twython
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import RegexpTokenizer
import numpy as np
import matplotlib.pyplot as plt
import codecs
import json


class SenTweet:

    def __init__(self, api_key, api_secret_key):
        self.twitter = Twython(api_key, api_secret_key)
        self.tokenizer = RegexpTokenizer('\w+')

    def twitter_search(self, q, count, result_type='recent'):
        query = {
                'q': q,  
                'result_type': result_type, # 'recent', 'popular', 'mixed'
                 'count': count,
                 'lang': 'en',
                 }
        
        res = self.twitter.search(**query)
        ###
        rate = self.twitter.get_lastfunction_header('x-rate-limit-remaining')
        print(rate)
        ###
        return res

    def sentiment(self, text, eps=1e-3):
        tokens = [unicode.lower(x) for x in self.tokenizer.tokenize(text)]
        pscore = 0
        nscore = 0
        
        for t in tokens:
            synsets = swn.senti_synsets(t)
            if len(synsets) > 0:
                scores = [(s.pos_score(), s.neg_score()) for s in synsets]
                pos, neg = np.mean([s[0] for s in scores]), np.mean([s[1] for s in scores])
                pscore += pos
                nscore += neg
        
        pscore += eps
        nscore += eps
        norm = float(pscore + nscore) if pscore + nscore > 0 else 1
        return (pscore/norm, nscore/norm)
    
    def stats(self, tweets, pos_neg_min_ratio):
        post = 0
        negt = 0
        
        for tw in tweets['statuses']:
            ps, ns = self.sentiment(tw['text'])
            if ps/ns >= pos_neg_min_ratio:
                post += 1
            elif ns/ps >= pos_neg_min_ratio:
                negt += 1
        
        norm = float(post + negt) if post + negt > 0 else 1
        return (post/norm, negt/norm)


if __name__ == "__main__":
    twitter_api = {}
    f = codecs.open('auth.json', 'r')
    twitter_api = json.load(f)
    f.close()
    
    queries = ['linux', 'windows']
#     queries = ['android', 'ios']
#     queries = ['samsung', 'huawei', 'iphone', 'xiaomi']
#     queries = ['samsung', 'huawei', 'apple', 'xiaomi']
#     queries = ['rock', 'pop', 'rap']
#     queries = ['heavy metal', 'trash metal', 'death metal']
#     queries = ['volkswagen', 'dacia']
#     queries = ['graduation']
    
    N = 1000
    
    st = SenTweet(twitter_api['API_KEY'], twitter_api['API_SECRET_KEY'])
    res = [st.twitter_search(q=q, count=N, result_type='mixed') for q in queries]
        
    stats = [st.stats(r, pos_neg_min_ratio=2) for r in res]
    print(stats)
    
    bar_width = 0.5
    plt.bar(range(len(stats)), [stats[i][0] for i in range(len(stats))], bar_width, color='green')
    plt.bar(range(len(stats)), [stats[i][1] for i in range(len(stats))], bar_width, bottom=[stats[i][0] for i in range(len(stats))], color='red')
    plt.xticks(range(len(stats)), queries, rotation='vertical')
    plt.show()
    