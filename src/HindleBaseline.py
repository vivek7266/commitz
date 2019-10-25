import glob
import os
import string
import re

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from config import Metrics, VectorizerConfig, LdaConfig, SvmConfig

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
lemmatizer = WordNetLemmatizer()
rt = RegexpTokenizer(r'[^\W_]+|[^\W_\s]+')
# stopset = set(stopwords.words('english'))
stopset = {"the", "for", "and", "are", "were", "does", "has", "had", "did", "with", "into", "org", "svn", "cvs", "from",
           "this", "not", "more", "that"}

data_path = "/Users/saurabh/Downloads/ncsu/study/thesis/project/data/"

corrective_keys = {"fix", "bug", "problem", "incorrect", "correct", "error", "fixup", "fail"}


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


def set_lda_topics_in_df(X_tf, lda, no_topics, df):
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


def baseline(train_df, num_features=VectorizerConfig.NUM_FEATURES, num_topics=LdaConfig.NUM_TOPICS,
             num_iter=LdaConfig.NUM_ITERATIONS):
    tf_vectorizer = get_tf_vectorizer(num_features)

    X_tf, feature_names = vectorize(train_df['msg_str'], tf_vectorizer)
    lda = fit_lda(X_tf, num_topics, num_iter)
    processed_df = set_lda_topics_in_df(X_tf, lda, num_topics, train_df)
    topic_cols = get_topic_cols(processed_df)
    return tf_vectorizer, topic_cols, processed_df, lda, feature_names


def get_topic_top_words(model, feature_names, no_top_words=LdaConfig.NUM_TOP_WORDS):
    topic_top_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_top_words.append(top_words)
    return topic_top_words


def map_topic_to_bugginess(topic_top_words):
    max_i = 0
    for i in range(len(topic_top_words)):
        len_top_i = len(set(topic_top_words[i]) & corrective_keys)
        if len_top_i > max_i:
            max_i = i
    print("Topic_{} is buggy".format(max_i))
    return max_i


def pred_topic_bugginess(row, topic_cols, buggy_index):
    data = row[topic_cols].values
    if 2 * data[buggy_index] - np.sum(data) > 0:
        return 1
    else:
        return 0


def main():
    project_names = [['abinit'], ['libmesh'], ['mdanalysis']]
    # project_names = [['abinit']]
    metrics = {}
    for idx, p in enumerate(project_names):
        print("Playing with {}".format(p))
        project_key = '+'.join(p)
        raw_df = pre_process_text_dataframe(get_raw_df(p))
        train_df, test_df = train_test_split(raw_df, test_size=0.2)

        num_topics = 2

        # Training
        tf_vectorizer, topic_cols, processed_df, lda, feature_names = baseline(train_df, num_topics=num_topics)
        topic_top_words = get_topic_top_words(lda, feature_names)
        print(topic_top_words)
        buggy_topic = map_topic_to_bugginess(topic_top_words)

        # Testing
        X_tf_test = tf_vectorizer.transform(test_df['msg_str'])
        processed_test_df = set_lda_topics_in_df(X_tf_test, lda, num_topics, test_df)
        processed_test_df["pred_class"] = processed_test_df.apply(
            lambda x: pred_topic_bugginess(x, topic_cols, buggy_topic), axis=1)
        metrics[project_key] = get_cm_metrics(processed_test_df["buggy"], processed_test_df["pred_class"],
                                              key="Testing")
    return metrics


if __name__ == '__main__':
    f_name = "hindle-tmp.csv"
    final_mean_metrics = {}
    sum_metrics = {}
    run_metrics = {}
    for i in range(20):
        metrics_map = main()
        for project_key, metrics in metrics_map.items():
            if project_key not in run_metrics:
                run_metrics[project_key] = []
            run_metrics[project_key].append(metrics_map[project_key])
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
    print("\n Run Metrics: ")
    print(run_metrics)
