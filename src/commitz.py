import re, string, os, glob
import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
from config import LdaConfig, MetaConfig, Metrics, VectorizerConfig, SvmConfig

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
rt = RegexpTokenizer(r'[^\W_]+|[^\W_\s]+')
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language='english')
stopset = set(stopwords.words('english'))
model_top_map_cache = {}


# Andreas Mauckuza

def plot_metrices(metrics, img="metrics.png"):
    n_groups = 5
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8
    rects1 = plt.bar(index, metrics['abinit'], bar_width,
                     alpha=opacity,
                     color='b',
                     label='abinit')
    rects2 = plt.bar(index + bar_width, metrics['libmesh'], bar_width,
                     alpha=opacity,
                     color='r',
                     label='libmesh')
    rects3 = plt.bar(index + 2 * bar_width, metrics['mdanalysis'], bar_width,
                     alpha=opacity,
                     color='g',
                     label='mdanalysis')
    plt.xlabel('projects')
    plt.ylabel('Scores')
    plt.title('Metrics of Projects')
    plt.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall', 'F1', 'F2'))
    plt.legend()

    plt.tight_layout()
    plt.show()
    fig.savefig(img)
    plt.close(fig)


def get_file_names(projects):
    filenames = []
    for p in projects:
        s = "/Users/saurabh/workspace/fss/project/data/data-collection/labeled_commits/human/{}/".format(p)
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
    raw_df['tknz_msg'] = raw_df['message'] \
        .apply(rt.tokenize) \
        .apply(lambda tkns: [lemmatizer.lemmatize(w.lower()) for w in tkns])
    raw_df['msg'] = raw_df['tknz_msg'] \
        .apply(lambda tkns: list(filter(lambda word:
                                        word not in stopset
                                        and word not in string.punctuation
                                        and re.match(r'[^\W\d]*$', word)
                                        and len(word) > 2, tkns)))
    raw_df['msg_str'] = raw_df['msg'] \
        .apply(lambda tkns: ' '.join(tkns))
    return raw_df


def get_topic_details(model, feature_names, topic_indices, topic_word_prob, feature_names_set,
                      no_top_words=LdaConfig.NUM_TOP_WORDS):
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx not in topic_word_prob:
            topic_word_prob[topic_idx] = []
        top_features = [(feature_names[i], topic[i]) for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_word_prob[topic_idx] = top_features
        feature_set = set([val[0] for val in top_features])
        feature_names_set.update(feature_set)
        word_list_for_topic = [val[0] for val in top_features]
        word_probs_for_topic = [val[1] for val in top_features]
        bug_prob_sum = 0
        for bug_word in MetaConfig.BUGGY_KEYWORDS:
            if bug_word in word_list_for_topic:
                bug_prob_sum += word_probs_for_topic[word_list_for_topic.index(bug_word)]
        topic_indices.append(bug_prob_sum)

        print("Topic %d:" % (topic_idx))
        print(", ".join([val[0] for val in top_features]))
    return topic_indices, topic_word_prob, feature_names_set


def keyword_buggy(tokens):
    return 1 if any(token in MetaConfig.BUGGY_KEYWORDS for token in tokens) else 0


def get_topic_top_words(model, feature_names, no_top_words=LdaConfig.NUM_TOP_WORDS):
    if str(model) in model_top_map_cache:
        return model_top_map_cache[str(model)]
    topic_top_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_top_words.append(top_words)
    model_top_map_cache[str(model)] = topic_top_words
    return topic_top_words


def get_top_topics(words, model, feature_names, topic_indices):
    topic_ranks = []
    topic_top_words = get_topic_top_words(model, feature_names)
    for top_words in topic_top_words:
        topic_freq = 0
        for w in words:
            if w in top_words:
                topic_freq += 1
        topic_ranks.append(topic_freq)
    is_buggy_at_topic = 0
    max_val = max(topic_ranks)
    idx = topic_ranks.index(max_val)
    if idx in topic_indices:
        is_buggy_at_topic = 1
    return max_val, idx, is_buggy_at_topic


def doc_word_mapping(words, topic0, topic1, feature_names_list, topic0_word_prob_map, topic1_word_prob_map,
                     buggy_topic):
    is_doc_buggy_lda = 0
    is_max_0 = True
    topic_prob0 = topic0
    topic_prob1 = topic1
    if topic_prob1 > topic_prob0:
        is_max_0 = False
    if (is_max_0 and buggy_topic == 0) or (not is_max_0 and buggy_topic == 1):
        is_doc_buggy_lda = 1
    # weighted_words = [0] * (len(feature_names_list) + len(MetaConfig.BUGGY_KEYWORDS))
    # weighted_words = [0] * (len(feature_names_list))
    weighted_words = [0] * (len(feature_names_list) + 3)
    weighted_words[-1] = topic_prob1
    weighted_words[-2] = topic_prob0
    weighted_words[-3] = is_doc_buggy_lda
    uniq_words = Counter(words)
    for idx, w in enumerate(feature_names_list):
        count = 0
        if w in uniq_words:
            count = uniq_words[w]
        # if w in MetaConfig.BUGGY_KEYWORDS:
        #     idx = -1 * MetaConfig.BUGGY_KEYWORDS.index(w) + 1
        #     weighted_words[idx] = count * 1
        prob = 0
        if w in topic0_word_prob_map:
            prob += count * topic0_word_prob_map[w]
        if w in topic1_word_prob_map:
            prob += count * topic1_word_prob_map[w]
        # if is_max_0 and w in topic0_word_prob_map:
        #     prob += count * topic0_word_prob_map[w]
        # if not is_max_0 and w in topic1_word_prob_map:
        #     prob += count * topic1_word_prob_map[w]
        weighted_words[idx] = prob
    return weighted_words, is_doc_buggy_lda


def get_cm_metrics(y_true, y_test, key="unknown"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_test).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    beta = Metrics.F_BETA_BETA_VALUE
    f2 = (1 + np.power(beta, 2)) * prec * rec / (np.power(beta, 2) * prec + rec)
    print("What am I testing for: {}".format(key))
    print("accuracy", acc)
    print("precision", prec)
    print("recall", rec)
    print("f1", f1)
    print("f2", f2)
    print(tn, fp, fn, tp)
    print("\n\n")
    return acc, prec, rec, f1, f2


def train_with_voting_classifier(X_train, y_train, X=None, y=None):
    estimators = []
    clf1 = DecisionTreeClassifier(random_state=9)
    estimators.append(clf1)
    clf2 = SVC(C=100, kernel='linear', random_state=9)
    estimators.append(clf2)
    clf3 = SVC(C=100, kernel='rbf', gamma=0.01, random_state=9)
    estimators.append(clf3)
    eclf = VotingClassifier(estimators=[('nb', clf1), ('svml', clf2), ('svmr', clf3)], voting='hard')
    # for clf, label in zip([clf1, clf2, clf3, eclf], ['Naive Bayes', 'SVM Linear', 'SVM RBF', 'Ensemble']):
    #     scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
    #     print("F1 : %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    clf = eclf.fit(X_train, y_train)
    return clf


def train_with_linear_svm(X_train, y_train, c=SvmConfig.C_VALUE):
    clf = SVC(C=c, kernel='linear', random_state=9, max_iter=5000000)
    clf.fit(X_train, y_train)
    return clf


def train_with_multinomial_nb(X_train, y_train):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf


def train_with_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=9)
    # clf = RandomForestClassifier(n_estimators=100, random_state=9)
    clf.fit(X_train, y_train)
    return clf


def train_with_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=9)
    clf.fit(X_train, y_train)
    return clf


def vectorize(corpus, vectorizer):
    X_transformed = vectorizer.fit_transform(corpus)
    vectorized_feature_names = vectorizer.get_feature_names()
    return X_transformed, vectorized_feature_names


def fit_lda(X_tf, num_topics, num_iter):
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=num_iter, learning_method='online',
                                    learning_offset=50., random_state=9, evaluate_every=100).fit(X_tf)
    return lda


def get_tf_vectorizer(num_features):
    tf_vectorizer = CountVectorizer(max_features=num_features)
    # tf_vectorizer = CountVectorizer(ngram_range=(1, 3), max_df=0.95, min_df=2, max_features=num_features)
    return tf_vectorizer


def get_tfidf_vectorizer(num_features):
    tfidf_vectorizer = TfidfVectorizer(max_features=num_features)
    # tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.95, min_df=2, max_features=num_features)
    return tfidf_vectorizer


def transform_with_lda(lda, tf_feature_names, num_top_words):
    topic_indices, topic_word_prob, feature_names_set = get_topic_details(lda, tf_feature_names, [], {}, set(),
                                                                          num_top_words)
    feature_names_list = list(feature_names_set)
    print(topic_indices)
    buggy_topic = topic_indices.index(max(topic_indices))
    print(buggy_topic)
    print(topic_word_prob)
    print(feature_names_list)
    return topic_indices, feature_names_list, topic_word_prob, buggy_topic


def set_lda_topics_in_df(X_tf, lda, no_topics, df):
    lda_x = lda.transform(X_tf)
    for i in range(no_topics):
        topic_name = "Topic_{}".format(str(i))
        data = pd.Series(lda_x[:, i])
        df[topic_name] = data.values
    return df


def transform_df(raw_df, lda, feature_names, num_top_words):
    buggies = raw_df.groupby("buggy")["hash"].count()
    print("Buggy file counts:")
    print(buggies)
    # print(raw_df.shape)
    # print(raw_df.dtypes)

    topic_indices, feature_names_list, topic_word_prob, buggy_topic = transform_with_lda(lda, feature_names,
                                                                                         num_top_words)
    topic_index1 = 1
    topic_index0 = 0
    topic0_word_prob_map = dict(topic_word_prob[topic_index0])
    topic1_word_prob_map = dict(topic_word_prob[topic_index1])

    raw_df['topic_freq'], raw_df['topic_id'], raw_df['buggy_topic'] = zip(
        *raw_df['msg'].apply(lambda tkns: get_top_topics(tkns, lda, feature_names, topic_indices)))

    raw_df['word_prob'], raw_df['buggy_atlda'] = zip(*raw_df.apply(
        lambda x: doc_word_mapping(x['msg'], x['Topic_0'], x['Topic_1'], feature_names_list, topic0_word_prob_map,
                                   topic1_word_prob_map, buggy_topic), axis=1))

    # raw_df['word_prob'] = raw_df['word_prob'].apply(lambda x: np.array(x))

    tops_labels = raw_df.groupby(['topic_id', 'buggy']).size()
    tops_labels_topic = raw_df.groupby(['topic_id', 'buggy_atlda']).size()
    buggy_vs_buggyatlda = raw_df.groupby(['buggy', 'buggy_atlda']).size()
    print(tops_labels)
    print(tops_labels_topic)
    print(buggy_vs_buggyatlda)
    # for i, v in tops_labels.items():
    #     if i[0] in topic_indices:
    #         print('index: ', i, 'value: ', v)
    return raw_df


def voting_result(pred_y_1, pred_y_2):
    cumulative_prediction = []
    for i in range(len(pred_y_1)):
        if pred_y_1[i] == 1 or pred_y_2[i] == 1:
            cumulative_prediction.append(1)
        else:
            cumulative_prediction.append(0)
    return cumulative_prediction


def simple_linear_svm(final_df, X):
    y = final_df['buggy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9, stratify=y)
    print("baseline: # positive in train set: {}".format(len(y_train[y_train == 1])),
          "\nbaseline: # negative in train set: {}".format(len(y_train[y_train == 0])))
    print("baseline: # positive in test set: {}".format(len(y_test[y_test == 1])),
          "\nbaseline: # negative in test set: {}".format(len(y_test[y_test == 0])))
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    c = 50
    clf = train_with_linear_svm(X_train, y_train, c)
    pred_y = clf.predict(X_test)
    acc_clf, prec_clf, rec_clf, f1_clf, f2_clf = get_cm_metrics(y_test, pred_y, "validation set")
    return clf, scaler


def commitz_classifier(final_df, X_vectorized, mixmatch):
    # y_true = final_df['buggy']
    # y_lda = final_df['buggy_atlda']
    # acc_lda, prec_lda, rec_lda, f1_lda, f2_lda = get_cm_metrics(y_true, y_lda, "LDA")

    X = final_df['word_prob'].values.tolist()
    X = np.asarray(X)
    y = final_df['buggy']
    if mixmatch:
        X_vector_list = X_vectorized.toarray()
        X_final = np.concatenate((X, X_vector_list), axis=1)
    else:
        X_final = X

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.20, random_state=9, stratify=y)

    print("# positive in train set: {}".format(len(y_train[y_train == 1])),
          "\n# negative in train set: {}".format(len(y_train[y_train == 0])))
    print("# positive in test set: {}".format(len(y_test[y_test == 1])),
          "\n# negative in test set: {}".format(len(y_test[y_test == 0])))

    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Cs = [0.1, 10, 50, 100, 200, 400, 1000]
    c = 50
    clf = train_with_linear_svm(X_train, y_train, c)
    # clf = train_with_random_forest(X_train, y_train)
    # clf = train_with_voting_classifier(X_train, y_train)
    # clf = train_with_multinomial_nb(X_train, y_train)
    pred_y = clf.predict(X_test)
    acc_clf, prec_clf, rec_clf, f1_clf, f2_clf = get_cm_metrics(y_test, pred_y, "validation set")
    # print("c and f1 measure", c, f1_clf)
    # print("Best C value: {}".format(str(c)))
    return clf, scaler


def baseline_classifier(final_df, X):
    y = final_df['buggy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9, stratify=y)
    print("baseline: # positive in train set: {}".format(len(y_train[y_train == 1])),
          "\nbaseline: # negative in train set: {}".format(len(y_train[y_train == 0])))
    print("baseline: # positive in test set: {}".format(len(y_test[y_test == 1])),
          "\nbaseline: # negative in test set: {}".format(len(y_test[y_test == 0])))
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    c = 50
    clf = train_with_linear_svm(X_train, y_train, c)
    pred_y = clf.predict(X_test)
    acc_clf, prec_clf, rec_clf, f1_clf, f2_clf = get_cm_metrics(y_test, pred_y, "validation set")
    return clf, scaler


project_df_map = {}


def bugzy_main(num_features=VectorizerConfig.NUM_FEATURES, num_topics=LdaConfig.NUM_TOPICS,
               num_iter=LdaConfig.NUM_ITERATIONS,
               num_top_words=LdaConfig.NUM_TOP_WORDS, ensemble=False, mixmatch=True):
    # project_names = ['abinit', 'libmesh', 'lammps', 'mdanalysis']
    # project_names = [['abinit', 'libmesh', 'lammps', 'mdanalysis']]
    project_names = [['abinit'], ['libmesh'], ['mdanalysis']]
    metrics = {}
    for idx, p in enumerate(project_names):
        global model_top_map_cache
        model_top_map_cache = {}
        if str(p) not in project_df_map:
            raw_df = pre_process_text_dataframe(get_raw_df(p))
            project_df_map[str(p)] = raw_df
        else:
            raw_df = project_df_map[str(p)]

        test_raw_df = raw_df.sample(frac=0.2, replace=True)

        raw_train_df = raw_df[(raw_df['hash'].apply(lambda x: x not in test_raw_df['hash'].values))]

        tf_vectorizer = get_tf_vectorizer(num_features)
        tfidf_vectorizer = get_tfidf_vectorizer(num_features)
        X_tfidf = tfidf_vectorizer.fit_transform(raw_train_df['msg_str'])
        X, feature_names = vectorize(raw_train_df['msg_str'], tf_vectorizer)
        lda = fit_lda(X, num_topics, num_iter)
        processed_df = set_lda_topics_in_df(X, lda, num_topics, raw_train_df)
        final_df = transform_df(processed_df, lda, feature_names, num_top_words)
        clf, scaler = commitz_classifier(final_df, X_tfidf, mixmatch)

        X_test_vector = tf_vectorizer.transform(test_raw_df['msg_str'])
        X_tfidf_test_vector = tfidf_vectorizer.transform(test_raw_df['msg_str'])
        processed_test_df = set_lda_topics_in_df(X_test_vector, lda, num_topics, test_raw_df)
        test_df = transform_df(processed_test_df, lda, feature_names, num_top_words)
        y_test_true = test_df['buggy']

        # # test LDA
        # y_lda = final_df['buggy_atlda']
        # acc_lda, prec_lda, rec_lda, f1_lda, f2_lda = get_cm_metrics(y_test_true, y_lda, "LDA")

        X_test_raw = test_df['word_prob'].values.tolist()
        X_test_raw = np.asarray(X_test_raw)
        if mixmatch:
            X_test_vector_list = X_tfidf_test_vector.toarray()
            X_test_final = np.concatenate((X_test_raw, X_test_vector_list), axis=1)
        else:
            X_test_final = X_test_raw
        X_test = scaler.transform(X_test_final)
        y_test_pred = clf.predict(X_test)
        # If ensemble result is required
        if ensemble:
            clf_2, scaler_2 = simple_linear_svm(final_df, X.todense())
            X_test2 = scaler_2.transform(X_test_vector.todense())
            y_test_pred_2 = clf_2.predict(X_test2)
            y_test_pred_final = voting_result(y_test_pred, y_test_pred_2)
        else:
            y_test_pred_final = y_test_pred
        # acc_clf, prec_clf, rec_clf, f1_clf, f2_clf = get_cm_metrics(y_test_pred, y_test_true, "Test Set")
        acc_clf, prec_clf, rec_clf, f1_clf, f2_clf = get_cm_metrics(y_test_pred_final, y_test_true, "Test Set")

        metrics[idx] = [acc_clf, prec_clf, rec_clf, f1_clf, f2_clf]
    # plot_metrices(metrics, "tmp-metrics.png")
    print(metrics)
    return metrics


def bugzy_baseline(num_features=VectorizerConfig.NUM_FEATURES):
    project_names = [['abinit'], ['libmesh'], ['mdanalysis']]
    metrics = {}
    for idx, p in enumerate(project_names):
        global model_top_map_cache
        model_top_map_cache = {}
        if str(p) not in project_df_map:
            raw_df = pre_process_text_dataframe(get_raw_df(p))
            project_df_map[str(p)] = raw_df
        else:
            raw_df = project_df_map[str(p)]

        test_raw_df = raw_df.sample(frac=0.2, replace=True)

        raw_train_df = raw_df[(raw_df['hash'].apply(lambda x: x not in test_raw_df['hash'].values))]

        vectorizer = get_tf_vectorizer(num_features)
        X, feature_names = vectorize(raw_train_df['msg_str'], vectorizer)

        clf, scaler = baseline_classifier(raw_train_df, X.todense())

        X_test_vector = vectorizer.transform(test_raw_df['msg_str'])
        y_test_true = test_raw_df['buggy']
        X_test = scaler.transform(X_test_vector.todense())
        y_test_pred = clf.predict(X_test)
        acc_clf, prec_clf, rec_clf, f1_clf, f2_clf = get_cm_metrics(y_test_pred, y_test_true, "Test Set")
        metrics[idx] = [acc_clf, prec_clf, rec_clf, f1_clf, f2_clf]
    print(metrics)
    return metrics


def sig_test(n_iter=100, baseline=False):
    final_mean_metrics = {}
    sum_metrics = {}
    for i in range(n_iter):
        if baseline:
            metrics = bugzy_baseline()
        else:
            metrics = bugzy_main(ensemble=False, mixmatch=False)
        for k, v in metrics.items():
            if k not in sum_metrics:
                sum_metrics[k] = np.asarray(v)
            else:
                sum_metrics[k] = np.add(sum_metrics[k], np.asarray(v))
        mean_metrics = {}
        for k, v in sum_metrics.items():
            mean_metrics[k] = np.asarray(v) / (i + 1)
        print("\n Running mean metrics: {}".format(i))
        print(mean_metrics)
        final_mean_metrics = mean_metrics
    return final_mean_metrics


def driver_main(optimization=False, baseline=False):
    if baseline:
        return bugzy_baseline()
    if not optimization:
        return bugzy_main(ensemble=False, mixmatch=False)
    else:
        # num_features = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        # num_top_words = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        num_features = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        num_top_words = [5, 7, 9, 12, 15]
        parent_metrics = {}
        for ntw in num_top_words:
            for nf in num_features:
                if nf < 2 * ntw:
                    continue
                key = str(ntw) + "-" + str(nf)
                print("Working on :", key)
                parent_metrics[key] = bugzy_main(num_top_words=ntw, num_features=nf, ensemble=False)
        print(parent_metrics)
        return parent_metrics


if __name__ == '__main__':
    # driver_main(optimization=False, baseline=False)
    final_metrics = sig_test(n_iter=10, baseline=False)
    print("\n Final mean metrics:")
    print(final_metrics)
