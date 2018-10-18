import pandas as pd
import nltk
import re
from string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.tokenize import WhitespaceTokenizer, RegexpTokenizer, word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from tabulate import tabulate


lem = lem = WordNetLemmatizer()
tokenizer = WhitespaceTokenizer()
regex_tokenizer = RegexpTokenizer("[\w']+")
stop_words = set(stopwords.words('english') + list(string.ascii_lowercase) + ['xa'])

FILE = r'C:\Users\kumadee\Desktop\assignment1-find_top_keywords\NewsArticles_Top10Keywords.csv'

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv(FILE, parse_dates=['date'], index_col='date', date_parser=dateparse)

articles = []
tokens = []

import stopswods as st

stop_words = st.stop_words + list(string.punctuation)

def tokeniz(string):
    
    token = [t.lower() for t in regex_tokenizer.tokenize(string) if t.lower() not in stop_words ]
    # nltk.pos_tag(token)
    # bigram_finder = BigramCollocationFinder.from_words(token)
    # bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 1000)

    # for bigram_tuple in bigrams:
    #     x = "%s %s" % bigram_tuple
    #     token.append(x)        
    
    for t in token:
        # if t not in stop_words:
        tokens.append(lem.lemmatize(t))
        # tokens.append(t)
    
    
    
    articles.append(string)
    token.clear()
            
for article in range(df.shape[0]):
    # raw_article = re.sub('[^a-zA-Z]', ' ', df.title[article] + ' ' + df.content[article])
    raw_article = re.sub('[^a-zA-Z]', ' ', df.content[article])
    # raw_article = df.title[article] + ' ' + df.content[article]
    tokeniz(raw_article)


text = nltk.Text(tokens)
dist = FreqDist(tokens)
top_ten = dist.most_common(20)

top_ten = pd.DataFrame(dist.most_common(10), columns = ['Word', 'Count'])
top_ten.plot.bar(x='Word', y='Count')


a = nltk.pos_tag([a[0] for a in dist.most_common(1000)])

word = []
for k, pos in a:
    if pos not in  ['NN', 'IN', 'CD', 'JJ', 'JJS', 'VBN', 'VB', 'VBD']:
        word.append((k, dist.get(k)))
    
c = 0
for k,v in dist.most_common(20):
    
    print(k, "--->", v)
    c += 1
    if c == 5:
        break






all_chunks = []


def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    import itertools
    
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    print(tagged_sents, "-------------")
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower()
                  for key, group in itertools.groupby(
                          all_chunks, lambda chunk: chunk != 'O') if key]

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]


a = extract_candidate_chunks(articles)





articles = []
for article in range(df.shape[0]):
    # raw_article = re.sub('[^a-zA-Z]', ' ', df.title[article] + ' ' + df.content[article]) #Eliminating other than alphabet
    articles.append((df.title[article].lower() + ' ' + df.content[article].lower()))

raw_tokens = regex_tokenizer.tokenize(' '.join(articles))
bigram_finder = BigramCollocationFinder.from_words(raw_tokens)
bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 1000)
for bigram_tuple in bigrams:
    x = "%s %s" % bigram_tuple
    raw_tokens.append(x)

tokens = []
for t in raw_tokens:
    if t.lower() not in stop_words:
        tokens.append(t.lower()) 

text = nltk.Text(tokens)
dist = FreqDist(text)
top_ten = dist.most_common(10)










import RAKE

rake = RAKE.Rake(RAKE.SmartStopList())

rake.run(' '.join(articles), minCharacters = 3, maxWords = 2, minFrequency = 1)


        