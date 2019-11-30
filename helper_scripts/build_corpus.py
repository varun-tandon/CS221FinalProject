import pandas as pd

def main():
	print("here")
	df = pd.read_csv('../data_parsed.csv')
	build(df)

def build(df):

	corpus = open('corpus.txt', 'w')
	for index, row in df.iterrows():
		name = row['ScriptLink']
		with open('../movie_scripts/' + name) as file:
			script = file.readlines()
			script = "".join(script)
			corpus.write(repr(script))
	corpus.close()

if __name__ == '__main__':
	main()