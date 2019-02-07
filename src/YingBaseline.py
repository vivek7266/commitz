"""
Corrective: Bugfix bug cause error failure fix miss null warn wrong bad correct incorrect problem opps
valid invalid fail bad dump except

Adaptive: Add new create feature function appropriate available change compatibility config configuration text
current default easier future information internal method necessary old patch protocol provide release replace
require security simple structure switch context trunk useful user version install introduce faster init

Perfective: Clean cleanup consistent declaration definition documentation move prototype remove static style
unused variable whitespace header include dead inefficient useless

Use above term lists as signifier documents

Classification: distance between unclassified and signifier document


"""

import glob
import os
import re
import string
import re

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.config import Metrics, VectorizerConfig, LdaConfig

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ying_sig_corrective = {'bugfix', 'bug', 'cause', 'error', 'failure', 'fix', 'miss', 'null', 'warn', 'wrong', 'bad',
                       'correct', 'incorrect', 'problem', 'opps', 'valid', 'invalid', 'fail', 'bad', 'dump', 'except'}
ying_sig_adaptive = {'add', 'new', 'create', 'feature', 'function', 'appropriate', 'available', 'change',
                     'compatibility', 'config', 'configuration', 'text', 'current', 'default', 'easier', 'future',
                     'information', 'internal', 'method', 'necessary', 'old', 'patch', 'protocol', 'provide', 'release',
                     'replace', 'require', 'security', 'simple', 'structure', 'switch', 'context', 'trunk', 'useful',
                     'user', 'version', 'install', 'introduce', 'faster', 'init'}
ying_sig_perfective = {'clean', 'cleanup', 'consistent', 'declaration', 'definition', 'documentation', 'move',
                       'prototype', 'remove', 'static', 'style', 'unused', 'variable', 'whitespace', 'header',
                       'include', 'dead', 'inefficient', 'useless'}
signifier_df = pd.DataFrame(
    [' '.join(ying_sig_corrective), ' '.join(ying_sig_adaptive), ' '.join(ying_sig_perfective)],
    columns=["terms_list"])
signifier_df['labels'] = ["corrective", "adaptive", "perfective"]

lemmatizer = WordNetLemmatizer()
rt = RegexpTokenizer(r'[^\W_]+|[^\W_\s]+')
# stopset = set(stopwords.words('english'))
stopset = {"the"}

data_path = "/Users/saurabh/Downloads/ncsu/study/thesis/project/data/"


def get_cm_metrics(y_true, y_test, key="unknown", p=False):
    tn, fp, fn, tp = confusion_matrix(y_true, y_test).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    beta = Metrics.F_BETA_BETA_VALUE
    f2 = (1 + np.power(beta, 2)) * prec * rec / (np.power(beta, 2) * prec + rec)
    if p:
        print("What am I testing for: {}".format(key))
        print("accuracy", acc)
        print("precision", prec)
        print("recall", rec)
        print("f1", f1)
        print("f2", f2)
        print(tn, fp, fn, tp)
        print("\n\n")
    return acc, prec, rec, f1, f2


def get_tf_vectorizer(num_features):
    tf_vectorizer = CountVectorizer(max_features=num_features)
    return tf_vectorizer


def vectorize(corpus, vectorizer):
    X_transformed = vectorizer.fit_transform(corpus)
    vectorized_feature_names = vectorizer.get_feature_names()
    return X_transformed, vectorized_feature_names


def fit_lda(X_tf, num_topics, num_iter):
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=num_iter, learning_method='online',
                                    learning_offset=50., random_state=9, evaluate_every=100).fit(X_tf)
    return lda


def set_lda_topics_in_df(X_tf, lda, no_topics, df, sig_df, len_signifier):
    lda_x = lda.transform(X_tf)
    for i in range(no_topics):
        topic_name = "Topic_{}".format(str(i))
        data = pd.Series(lda_x[:-len_signifier, i])
        df[topic_name] = data.values
        sig_data = pd.Series(lda_x[-len_signifier:, i])
        sig_df[topic_name] = sig_data.values
    return df, sig_df


def set_lda_topics_in_test_df(X_tf, lda, no_topics, df):
    lda_x = lda.transform(X_tf)
    for i in range(no_topics):
        topic_name = "Topic_{}".format(str(i))
        data = pd.Series(lda_x[:, i])
        df[topic_name] = data.values
    return df


def get_topic_cols(df):
    rc = re.compile(r'Topic*')
    topic_list = list(filter(rc.search, df.columns))
    return topic_list


def similarity(row, sig_data, topic_cols):
    data = row[topic_cols].values
    max_dot = -1
    ret_idx = -1
    for i in range(sig_data.shape[0]):
        dot = np.dot(data, sig_data[i])
        if dot > max_dot:
            ret_idx = i
            max_dot = dot
    return ret_idx


def baseline(train_df, num_features=VectorizerConfig.NUM_FEATURES, num_topics=LdaConfig.NUM_TOPICS,
             num_iter=LdaConfig.NUM_ITERATIONS):
    tf_vectorizer = get_tf_vectorizer(num_features)

    # vectorize train dataframe merged with signifier data
    # print(train_df['msg_str'].shape, signifier_df['terms_list'].shape)
    X_tf, feature_names = vectorize(np.append(train_df['msg_str'], signifier_df['terms_list'], axis=0), tf_vectorizer)
    lda = fit_lda(X_tf, num_topics, num_iter)
    processed_df, processed_sig_df = set_lda_topics_in_df(X_tf, lda, num_topics, train_df, signifier_df,
                                                          signifier_df.shape[0])
    # print(processed_df.head(3), processed_sig_df.head(3))
    topic_cols = get_topic_cols(processed_sig_df)
    sig_topic_df = processed_sig_df.loc[:, topic_cols]
    # print(sig_topic_df.head(3))
    sig_data = sig_topic_df.values
    processed_df = process_similarity(processed_df, sig_data, topic_cols)
    get_cm_metrics(processed_df["buggy"], processed_df["pred_class"], key="Training")
    return tf_vectorizer, topic_cols, processed_df, lda, sig_data


def process_similarity(processed_df, sig_data, topic_cols):
    processed_df["classified"] = processed_df.apply(lambda row: similarity(row, sig_data, topic_cols), axis=1)
    # print(processed_df[["msg_str", "Topic_0", "Topic_1", "Topic_2", "buggy", "classified"]].head(5))
    processed_df["pred_class"] = processed_df["classified"].apply(lambda x: 1 if x == 0 else 0)
    return processed_df


def get_file_names(projects):
    base_path = "{root}data-collection/labeled_commits/human/{filename}/"
    filenames = []
    for p in projects:
        s = base_path.format(root=data_path, filename=p)
        path = s
        print(path)
        files = glob.glob(os.path.join(path, "*.csv"))
        filenames.extend(files)
    return filenames


def get_raw_df(projects):
    all_files = get_file_names(projects)
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    df = concatenated_df.drop_duplicates().reset_index()
    raw_df = df.dropna().reset_index()
    return raw_df


def pre_process_text_dataframe(raw_df):
    raw_df['tknz_msg'] = raw_df['message'].apply(rt.tokenize).apply(
        lambda tkns: [lemmatizer.lemmatize(w.lower()) for w in tkns])
    raw_df['msg'] = raw_df['tknz_msg'].apply(lambda tkns: list(filter(
        lambda word: word not in stopset and word not in string.punctuation and re.match(r'[^\W\d]*$', word) and len(
            word) > 2, tkns)))
    raw_df['msg_str'] = raw_df['msg'].apply(lambda tkns: ' '.join(tkns))
    return raw_df


def main():
    project_names = [['abinit'], ['libmesh'], ['mdanalysis']]
    # project_names = [['abinit']]
    metrics = {}
    for idx, p in enumerate(project_names):
        print("Playing with {}".format(p))
        project_key = '+'.join(p)
        raw_df = pre_process_text_dataframe(get_raw_df(p))
        train_df, test_df = train_test_split(raw_df, test_size=0.2)

        num_topics = LdaConfig.NUM_TOPICS

        # Training
        tf_vectorizer, topic_cols, processed_df, lda, sig_data = baseline(train_df, num_topics=num_topics)

        # Testing
        X_tf_test = tf_vectorizer.transform(test_df['msg_str'])
        processed_test_df = set_lda_topics_in_test_df(X_tf_test, lda, num_topics, test_df)
        processed_test_df = process_similarity(processed_test_df, sig_data, topic_cols)
        metrics[project_key] = get_cm_metrics(processed_test_df["buggy"], processed_test_df["pred_class"],
                                              key="Testing")
    return metrics


if __name__ == '__main__':
    f_name = "ying.csv"
    final_mean_metrics = {}
    sum_metrics = {}
    for i in range(20):
        metrics_map = main()
        for project_key, metrics in metrics_map.items():
            if project_key not in sum_metrics:
                sum_metrics[project_key] = np.asarray(metrics)
            else:
                sum_metrics[project_key] = np.add(sum_metrics[project_key], np.asarray(metrics))

        # Print running mean metrics
        mean_metrics = {}
        for key, sum_till in sum_metrics.items():
            mean_metrics[key] = np.asarray(sum_till) / (i + 1)
        print("\n Running mean metrics: {}".format(i))
        print(mean_metrics)
        final_mean_metrics = mean_metrics
    metrics_df = pd.DataFrame.from_dict(final_mean_metrics, orient='index',
                                        columns=["accuracy", "precision", "recall", "f1", "f2"])
    metrics_df.to_csv("{}".format(f_name), index=True, header=True, index_label="project")
    print(metrics_df.head(10))
