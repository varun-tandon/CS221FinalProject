#trying out doc2vec
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import pandas as pd

class MyCorpus:
    def __init__(self, train_data):
        self.train_data = train_data
        
    def __iter__(self):
        p = PorterStemmer()
        for index, row in self.train_data.iterrows():
            name = row['ScriptLink']
            with open('./movie_scripts/' + name) as file:
                #print("Im here")
                script = file.readlines()
                script = "".join(script)
                script = remove_stopwords(script)
                script = p.stem_sentence(script)
                words = simple_preprocess(script)
                yield TaggedDocument(words=words, tags=[index])

class Doc2VecTrainer:
    def __init__(self, train_corpus):
        self.train_corpus = train_corpus

    def run(self):
        print('app started')

        cores = multiprocessing.cpu_count()
        print('num of cores is %s' % cores)
        gc.collect()
        load_existing = False
        if load_existing:
            print('loading an exiting model')
            model = Doc2Vec.load(PATH_TO_EXISTING_MODEL)
        else:
            print('reading training corpus from %s' % self.train_corpus)
            corpus_data = MyCorpus(self.train_corpus)
            
            model_dimensions = 100
            model = Doc2Vec(vector_size=model_dimensions, window=10, min_count=3, sample=1e-4, negative=5, workers=cores, dm=1)
            print('building vocabulary...')
            model.build_vocab(corpus_data)

            model.train(corpus_data, total_examples=model.corpus_count, epochs=20)
            model.save(doc2vec_model)
            model.save_word2vec_format(word2vec_model)

        print('total docs learned %s' % (len(model.docvecs)))

def main():
    df = pd.read_csv('data_parsed.csv')
    doc2vec_model = Doc2VecTrainer(df)
    doc2vec_model.run()

if __name__ == '__main__':
    main()



