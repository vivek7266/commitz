#!/usr/bin/env python
# coding: utf-8

from collections import Counter

import numpy as np
import pandas as pd
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import re

from config import VectorizerConfig, LdaConfig, Metrics, SvmConfig

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sto_datapath = "/Users/saurabh/Downloads/ncsu/study/thesis/datasets/Classified_SE_preprocessed/"
num_classes = 10
name_classes = ["academia", "cs", "diy", "expressionengine", "judaism", "photo", "rpg", "scifi", "ux", "webmasters"]
fnames = []
for i in range(0, num_classes):
    fnames.append("{}SE{}.txt".format(sto_datapath, str(i)))

raw_dfs = []
for nc in name_classes:
    raw_fname = "{}{}.txt".format(sto_datapath, nc)
    raw_df = pd.read_csv(raw_fname, delimiter='>>>', header=None, index_col=None)
    raw_df.columns = ["content", "tags"]
    raw_df['label'] = nc
    raw_dfs.append(raw_df)
raw_dfs = pd.concat(raw_dfs, ignore_index=True)
raw_input_df = raw_dfs.drop_duplicates()
raw_input_df = raw_input_df.dropna().reset_index()
# print(raw_input_df.sample(10))
print(raw_input_df.shape)


# print(fnames)


def visulaize_lda(lda_model, transformed_x, vectorizer):
    pyLDAvis.sklearn.prepare(lda_model, transformed_x, vectorizer)
    # pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer, mds='tsne')


def get_topic_details(model, feature_names, num_top_words, topic_word_prob, feature_names_set):
    for topic_idx, topic in enumerate(model.components_):
        #         print("Topic %d:" % (topic_idx))
        if topic_idx not in topic_word_prob:
            topic_word_prob[topic_idx] = []
        top_features = [(feature_names[i], topic[i]) for i in topic.argsort()[:-num_top_words - 1:-1]]
        topic_word_prob[topic_idx] = top_features
        #         print(" ".join(top_features))
        feature_set = set([val[0] for val in top_features])
        feature_names_set.update(feature_set)
        print("Topic %d:" % (topic_idx))
        print(", ".join([val[0] for val in top_features]))
    return topic_word_prob, feature_names_set


def transform_with_lda(lda, feature_names, num_top_words):
    topic_word_prob = {}
    feature_names_set = set()
    topic_word_prob, feature_names_set = get_topic_details(lda, feature_names, num_top_words, topic_word_prob,
                                                           feature_names_set)
    feature_names_list = list(feature_names_set)

    # print(topic_word_prob)
    # print(feature_names_list)

    return topic_word_prob, feature_names_list


def set_lda_topics_in_df(X_tf, lda, no_topics, df):
    lda_x = lda.transform(X_tf)
    df["topic_probs"] = [each for each in lda_x.tolist()]
    for i in range(no_topics):
        topic_name = "Topic_{}".format(str(i))
        data = pd.Series(lda_x[:, i])
        df[topic_name] = data.values
    return df


def get_topic_cols(df):
    rc = re.compile(r'Topic*')
    topic_list = list(filter(rc.search, df.columns))
    return topic_list


def doc_word_mapping_v2(words, topic0_word_prob_map, topic1_word_prob_map, feature_names_list, topic_probs):
    weighted_words = [0] * (len(feature_names_list))
    weighted_words.extend(topic_probs)
    uniq_words = Counter(words.split(' '))
    for idx, w in enumerate(feature_names_list):
        count = 0
        if w in uniq_words:
            count = uniq_words[w]
        prob = 0
        if w in topic0_word_prob_map:
            prob += count * topic0_word_prob_map[w]
        if w in topic1_word_prob_map:
            prob += count * topic1_word_prob_map[w]
        weighted_words[idx] = prob
    return weighted_words


def doc_word_mapping(words, topic0, topic1, topic0_word_prob_map, topic1_word_prob_map, feature_names_list):
    is_max_0 = True
    topic_prob0 = topic0
    topic_prob1 = topic1
    if topic_prob1 > topic_prob0:
        is_max_0 = False
    weighted_words = [0] * (len(feature_names_list) + 2)
    weighted_words[-1] = topic_prob1
    weighted_words[-2] = topic_prob0
    uniq_words = Counter(words.split(' '))
    for idx, w in enumerate(feature_names_list):
        count = 0
        if w in uniq_words:
            count = uniq_words[w]
        prob = 0
        if is_max_0 and w in topic0_word_prob_map:
            prob += count * topic0_word_prob_map[w]
        if not is_max_0 and w in topic1_word_prob_map:
            prob += count * topic1_word_prob_map[w]
        weighted_words[idx] = prob
    return weighted_words


def transform_df(raw_df, lda, feature_names, num_top_words):
    topic_word_prob, feature_names_list = transform_with_lda(lda, feature_names, num_top_words)

    topic_index1 = 1
    topic_index0 = 0
    topic0_word_prob_map = dict(topic_word_prob[topic_index0])
    topic1_word_prob_map = dict(topic_word_prob[topic_index1])
    # raw_df['word_prob'] = raw_df.apply(
    #     lambda x: doc_word_mapping(x['content'], x['Topic_0'], x['Topic_1'], topic0_word_prob_map, topic1_word_prob_map,
    #                                feature_names_list), axis=1)
    raw_df['word_prob'] = raw_df.apply(
        lambda x: doc_word_mapping_v2(x['content'], topic0_word_prob_map, topic1_word_prob_map,
                                      feature_names_list, topic_probs=x['topic_probs']), axis=1)
    final_df = raw_df[["word_prob", "label_code"]]
    return final_df, topic_word_prob, feature_names_list


def transform_test_df(raw_df, topic_word_prob, feature_names_list):
    topic_index1 = 1
    topic_index0 = 0
    topic0_word_prob_map = dict(topic_word_prob[topic_index0])
    topic1_word_prob_map = dict(topic_word_prob[topic_index1])

    # raw_df['word_prob'] = raw_df.apply(
    #     lambda x: doc_word_mapping(x['content'], x['Topic_0'], x['Topic_1'], topic0_word_prob_map, topic1_word_prob_map,
    #                                feature_names_list), axis=1)
    raw_df['word_prob'] = raw_df.apply(
        lambda x: doc_word_mapping_v2(x['content'], topic0_word_prob_map, topic1_word_prob_map,
                                      feature_names_list, topic_probs=x['topic_probs']), axis=1)
    final_df = raw_df[["word_prob", "label_code"]]
    return final_df


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


def get_x_y(final_df):
    X = final_df['word_prob'].values.tolist()
    y = final_df['label_code']
    return X, y


def start_cooking(final_df):
    X, y = get_x_y(final_df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=9, stratify=y)
    # print("# positive in train set: {}".format(len(y_train[y_train == 1])),
    #       "\n# negative in train set: {}".format(len(y_train[y_train == 0])))
    # print("# positive in test set: {}".format(len(y_val[y_val == 1])),
    #       "\n# negative in test set: {}".format(len(y_val[y_val == 0])))

    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    clf = train_with_random_forest(X_train, y_train)
    y_pred = clf.predict(X_val)
    get_cm_metrics(y_val, y_pred, "validation set", p=False)
    return clf, scaler


def train_with_linear_svm(X_train, y_train, c=SvmConfig.C_VALUE):
    clf = SVC(C=c, kernel='linear', random_state=9)
    clf.fit(X_train, y_train)
    return clf


def train_with_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=9)
    clf.fit(X_train, y_train)
    return clf


def fit_lda(X_tf, num_topics, num_iter):
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=num_iter, learning_method='online',
                                    learning_offset=50., random_state=9, evaluate_every=100).fit(X_tf)
    return lda


def get_vectorizer(num_features):
    # vectorizer = CountVectorizer(max_features=num_features)
    # vectorizer = CountVectorizer(ngram_range=(1, 3), max_df=0.95, min_df=2, max_features=num_features)
    # vectorizer = TfidfVectorizer(max_features=num_features)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.95, min_df=2, max_features=num_features)
    return vectorizer


def vectorize(corpus, vectorizer):
    X_transformed = vectorizer.fit_transform(corpus)
    vectorized_feature_names = vectorizer.get_feature_names()
    return X_transformed, vectorized_feature_names


def get_raw_df(fname=None, cache=True, cache_class=None):
    if cache:
        raw_df_class = raw_input_df[raw_input_df['label'] == cache_class]
        raw_df_non_class = raw_input_df[raw_input_df['label'] != cache_class]
        pos_size = len(raw_df_class)
        neg_size = len(raw_df_non_class)
        # raw_df = raw_df_non_class.sample(frac=10 * pos_size / neg_size, replace=True)
        raw_df = raw_df_non_class
        raw_df = pd.concat([raw_df_class, raw_df]).sample(frac=1)
        raw_df["label"] = raw_df["label"].apply(lambda val: "yes" if cache_class == str(val.strip()) else "no")
    else:
        raw_df = pd.read_csv(fname, delimiter='>>>', header=None, index_col=None)
        raw_df.columns = ["content", "label"]
    raw_df['idx'] = raw_df.index
    raw_df = raw_df.dropna().reset_index()
    raw_df["label"] = raw_df["label"].apply(lambda x: x.strip())

    raw_df["label_code"] = raw_df["label"].apply(lambda val: 1. if "yes" == str(val.strip()) else 0.)
    label_counts = raw_df.groupby("label")["idx"].count()
    label_code_counts = raw_df.groupby("label_code")["idx"].count()
    print(label_code_counts)
    return raw_df


def ingredients(train_df, num_features=VectorizerConfig.NUM_FEATURES, num_topics=LdaConfig.NUM_TOPICS,
                num_iter=LdaConfig.NUM_ITERATIONS):
    tf_vectorizer = get_vectorizer(num_features)

    X_tf, feature_names = vectorize(train_df['content'], tf_vectorizer)
    lda = fit_lda(X_tf, num_topics, num_iter)
    processed_df = set_lda_topics_in_df(X_tf, lda, num_topics, train_df)
    topic_cols = get_topic_cols(processed_df)
    return tf_vectorizer, topic_cols, processed_df, lda, feature_names


project_df_map = {}


def sto_lang_model(exp_class):
    class_key = exp_class
    metrics = {}
    # fname = fnames[class_key]
    # raw_df = get_raw_df(fname)

    if str(class_key) not in project_df_map:
        raw_df = get_raw_df(cache=True, cache_class=class_key)
        project_df_map[str(class_key)] = raw_df
    else:
        raw_df = project_df_map[str(class_key)]

    train_df, test_df = train_test_split(raw_df, test_size=0.2)

    # add all that can be added to train df as label 1
    train_df_pos = train_df[train_df["label_code"] == 1.]
    pos_size = train_df_pos.shape[0]
    pos_all = train_df.shape[0]

    if pos_size / pos_all <= 0.5:
        train_df_neg = train_df[train_df["label_code"] == 0.].sample(frac=2 * pos_size / (pos_all - pos_size),
                                                                     replace=True)
        train_df = pd.concat([train_df_pos, train_df_neg]).sample(frac=1)

    label_code_counts = train_df.groupby("label_code")["idx"].count()
    print("Train", label_code_counts)
    label_code_counts = test_df.groupby("label_code")["idx"].count()
    print("Test", label_code_counts)

    # Training
    num_topics = LdaConfig.NUM_TOPICS
    num_top_words = LdaConfig.NUM_TOP_WORDS_LARGE
    tf_vectorizer, topic_cols, processed_df, lda, feature_names = ingredients(train_df, num_topics=num_topics)
    final_df, topic_word_prob, feature_names_list = transform_df(processed_df, lda, feature_names, num_top_words)
    clf, scaler = start_cooking(final_df)

    # Testing
    X_tf_test = tf_vectorizer.transform(test_df['content'])
    processed_test_df = set_lda_topics_in_df(X_tf_test, lda, num_topics, test_df)
    processed_test_df = transform_test_df(processed_test_df, topic_word_prob, feature_names_list)
    test_X, test_y = get_x_y(processed_test_df)
    processed_test_df["pred_class"] = clf.predict(test_X)
    metrics[class_key] = get_cm_metrics(test_y, processed_test_df["pred_class"],
                                        key="Testing", p=True)
    return metrics


def main(exp_class):
    return sto_lang_model(exp_class)


if __name__ == '__main__':
    f_name = "tlmodel.csv"
    # exp_classes = [i for i in range(1, 10)]
    # exp_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    exp_classes = name_classes
    mean_metrics = {}
    sum_metrics = {}
    for c in exp_classes:
        for i in range(1):
            metrics_map = main(c)
            if c not in sum_metrics:
                sum_metrics[c] = np.asarray(metrics_map[c])
            else:
                sum_metrics[c] = np.add(sum_metrics[c], np.asarray(metrics_map[c]))

            # Print running mean metrics
            mean_metrics[c] = np.asarray(sum_metrics[c]) / (i + 1)
            print("\n Running mean metrics: {}".format(i))
            print(mean_metrics)
    metrics_df = pd.DataFrame.from_dict(mean_metrics, orient='index',
                                        columns=["accuracy", "precision", "recall", "f1", "f2"])
    print(metrics_df.head(10))
    metrics_df.to_csv("{}".format(f_name), index=True, header=True, index_label="project")
