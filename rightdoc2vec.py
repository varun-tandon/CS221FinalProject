from google.colab import drive
#drive.mount('/movie_scripts')
drive.mount("/content/gdrive", force_remount=True)

import multiprocessing
import gc

#trying out doc2vec
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import pandas as pd

class MyCorpus(object):
    def __init__(self, train_data):
        self.train_data = train_data
        
    def __iter__(self):
        p = PorterStemmer()
        counter = 0
        for index, row in self.train_data.iterrows():
            name = row['ScriptLink']
            #'/movie_scripts/My Drive/doc2vec_scripts/'
            try:
              with open('/content/gdrive/My Drive/doc2vec_scripts/' + name) as file:
                  #print("Im here")
                  script = file.readlines()
                  script = "".join(script)
                  script = remove_stopwords(script)
                  script = p.stem_sentence(script)
                  words = simple_preprocess(script)
                  yield TaggedDocument(words=words, tags=[index])
        #     except FileNotFoundError:
            except IOError:
                  counter += 1
                  #print('didnt find the file ' + name)
        print("not found:", counter)
        
class Doc2VecTrainer(object):
    def __init__(self, train_corpus):
        self.train_corpus = train_corpus

    def run(self):
        print('app started')

        cores = multiprocessing.cpu_count()
        #print(f'num of cores is {cores}')
        print('num of cores is %s', cores)
        gc.collect()
        load_existing = False
        
        if load_existing:
            print('loading an exiting model')
            model = Doc2Vec.load(PATH_TO_EXISTING_MODEL)
        else:
            #print(f'reading training corpus from {self.train_corpus}')
            print('reading training corpus from %s', self.train_corpus)
            corpus_data = MyCorpus(self.train_corpus)
            
            model_dimensions = 100
            model = Doc2Vec(vector_size=model_dimensions, window=10, min_count=3, sample=1e-4, negative=5, workers=cores, dm=1)
            print('building vocabulary...')
            model.build_vocab(corpus_data)

            model.train(corpus_data, total_examples=model.corpus_count, epochs=20)
            
            model.save(doc2vec_model)
            model.save_word2vec_format(word2vec_model)

        #print(f'total docs learned {len(model.docvecs)}')
        print('total docs learned %s', len(model.docvecs))



#corpus = MyCorpus()
#for vector in corpus:  # load one vector into memory at a time
   # print(vector)

    
#df = pd.read_csv('/movie_scripts/My Drive/doc2vec_scripts/data_parsed.csv')
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/varun-tandon/CS221FinalProject/master/data_parsed.csv?token=AHDIPXV4DXUX3G34RA26EFC55QZXK')
doc2vec_model = Doc2VecTrainer(df)


doc2vec_model.run()

