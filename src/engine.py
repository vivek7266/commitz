import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
import string

fname = "/Users/saurabh/workspace/fss/project/data/data-collection/labeled_commits/auto/abinit_concat.csv"
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main():
    lemmatizer = WordNetLemmatizer()

    raw_df = pd.read_csv(fname, index_col=0)
    buggies = raw_df.groupby("buggy")["hash"].count()
    print(buggies)

    raw_df['tknz_msg'] = raw_df['message'].apply(wordpunct_tokenize)
    print(raw_df['tknz_msg'].head(5))

    # raw_df['lmtz_msg'] = raw_df['tknz_msg'].apply(lambda tkns: [lemmatizer.lemmatize(t) for t in tkns])
    raw_df['msg'] = raw_df['tknz_msg'].apply(lambda tkns: list(
        filter(lambda word: word not in stopwords.words('english') and word not in string.punctuation, tkns)))
    print(raw_df['msg'].head(5))

    words = raw_df['msg'].apply(pd.Series).stack().drop_duplicates().tolist()
    print(len(words))
    print(words[:3])

    print(raw_df.dtypes)


if __name__ == '__main__':
    main()
