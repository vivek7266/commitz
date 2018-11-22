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
import matplotlib.pyplot as plt
from src import config

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
rt = RegexpTokenizer(r'[^\W_]+|[^\W_\s]+')
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language='english')
stopset = set(stopwords.words('english'))


def plot_metrices(metrics, img="metrics.png"):
    accs = []
    precs = []
    recs = []
    f1s = []
    f2s = []
    for k, v in metrics.items():
        if k == 'lammps':
            continue
        accs.append(k[0])
        precs.append(k[1])
        recs.append(k[2])
        f1s.append(k[3])
        f2s.append(k[4])
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
    rects2 = plt.bar(index+bar_width, metrics['libmesh'], bar_width,
                     alpha=opacity,
                     color='r',
                     label='libmesh')
    rects3 = plt.bar(index+2*bar_width, metrics['mdanalysis'], bar_width,
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


def get_file_names(project):
    filenames = []
    s = "/Users/saurabh/workspace/fss/project/data/data-collection/labeled_commits/human/{}/".format(project)
    path = s
    print(path)
    files = glob.glob(os.path.join(path, "*.csv"))
    filenames.extend(files)
    return filenames


def get_raw_df(project):
    all_files = get_file_names(project)
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
                      no_top_words=config.LdaConfig.NUM_TOP_WORDS):
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
        for bug_word in config.BUGGY_KEYWORDS:
            if bug_word in word_list_for_topic:
                bug_prob_sum += word_probs_for_topic[word_list_for_topic.index(bug_word)]
        topic_indices.append(bug_prob_sum)

        print("Topic %d:" % (topic_idx))
        print(", ".join([val[0] for val in top_features]))
    return topic_indices, topic_word_prob, feature_names_set


model_top_map_cache = {}


def get_topic_top_words(model, feature_names, no_top_words=config.LdaConfig.NUM_TOP_WORDS):
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
    buggy_topic = 0
    max_val = max(topic_ranks)
    idx = topic_ranks.index(max_val)
    if idx in topic_indices:
        buggy_topic = 1
    return max_val, idx, buggy_topic


def doc_word_mapping(words, topic0, topic1, feature_names_list, topic0_word_prob_map, topic1_word_prob_map):
    is_max_0 = True
    topic_prob0 = topic0
    topic_prob1 = topic1
    if topic_prob1 > topic_prob0:
        is_max_0 = False
    weighted_words = [0] * (len(feature_names_list) + 3)
    weighted_words[-2] = topic_prob1
    weighted_words[-3] = topic_prob0
    uniq_words = Counter(words)
    for idx, w in enumerate(feature_names_list):
        count = 0
        if w in uniq_words:
            count = uniq_words[w]
        if w in config.BUGGY_KEYWORDS:
            weighted_words[-1] = count * 1
        prob = 0
        if is_max_0 and w in topic0_word_prob_map:
            prob += count * topic0_word_prob_map[w]
        if not is_max_0 and w in topic1_word_prob_map:
            prob += count * topic1_word_prob_map[w]
        weighted_words[idx] = prob
    return weighted_words


def get_cm_metrics(y_true, y_test, key="unknown"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_test).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    beta = config.Metrics.F_BETA_BETA_VALUE
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
    clf1 = MultinomialNB()
    estimators.append(clf1)
    clf2 = SVC(C=100, kernel='linear')
    estimators.append(clf2)
    clf3 = SVC(C=100, kernel='rbf', gamma=0.01)
    estimators.append(clf3)
    eclf = VotingClassifier(estimators=[('nb', clf1), ('svml', clf2), ('svmr', clf3)], voting='hard')
    # for clf, label in zip([clf1, clf2, clf3, eclf], ['Naive Bayes', 'SVM Linear', 'SVM RBF', 'Ensemble']):
    #     scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
    #     print("F1 : %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    clf = eclf.fit(X_train, y_train)
    return clf


def train_with_linear_svm(X_train, y_train):
    clf = SVC(C=100, kernel='linear', random_state=9)
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


def get_vectorizer(num_features):
    # tfidf_vectorizer = TfidfVectorizer(max_features=no_features)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.95, min_df=2, max_features=num_features)
    tf_vectorizer = CountVectorizer(max_features=num_features)
    # tf_vectorizer = CountVectorizer(ngram_range=(1, 3), max_df=0.95, min_df=2, max_features=no_features)
    return tf_vectorizer


def transform_with_lda(lda, tf_feature_names):
    topic_indices, topic_word_prob, feature_names_set = get_topic_details(lda, tf_feature_names, [], {}, set())
    feature_names_list = list(feature_names_set)
    print(topic_indices)
    print(topic_indices.index(max(topic_indices)))
    print(topic_word_prob)
    print(feature_names_list)
    return topic_indices, feature_names_list, topic_word_prob


def set_lda_topics_in_df(X_tf, lda, no_topics, df):
    lda_x = lda.transform(X_tf)
    for i in range(no_topics):
        topic_name = "Topic_{}".format(str(i))
        data = pd.Series(lda_x[:, i])
        df[topic_name] = data.values
    return df


def transform_df(raw_df, lda, feature_names):
    buggies = raw_df.groupby("buggy")["hash"].count()
    print("Buggy file counts:")
    print(buggies)
    # print(raw_df.shape)
    # print(raw_df.dtypes)

    topic_indices, feature_names_list, topic_word_prob = transform_with_lda(lda, feature_names)
    topic_index1 = 1
    topic_index0 = 0
    topic0_word_prob_map = dict(topic_word_prob[topic_index0])
    topic1_word_prob_map = dict(topic_word_prob[topic_index1])

    raw_df['topic_freq'], raw_df['topic_id'], raw_df['buggy_topic'] = zip(
        *raw_df['msg'].apply(lambda tkns: get_top_topics(tkns, lda, feature_names, topic_indices)))

    tops_labels = raw_df.groupby(['topic_id', 'buggy']).size()
    print(tops_labels)
    for i, v in tops_labels.items():
        if i[0] in topic_indices:
            print('index: ', i, 'value: ', v)

    raw_df['word_prob'] = raw_df.apply(
        lambda x: doc_word_mapping(x['msg'], x['Topic_0'], x['Topic_1'], feature_names_list, topic0_word_prob_map,
                                   topic1_word_prob_map), axis=1)
    return raw_df


def commitz_classifier(final_df):
    y_true = final_df['buggy']
    y_lda = final_df['buggy_topic']
    acc_lda, prec_lda, rec_lda, f1_lda, f2_lda = get_cm_metrics(y_true, y_lda, "LDA")

    X = final_df['word_prob'].values.tolist()
    y = final_df['buggy']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9, stratify=y)

    print("# positive in train set: {}".format(len(y_train[y_train == 1])),
          "\n# negative in train set: {}".format(len(y_train[y_train == 0])))
    print("# positive in test set: {}".format(len(y_test[y_test == 1])),
          "\n# negative in test set: {}".format(len(y_test[y_test == 0])))

    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = train_with_linear_svm(X_train, y_train)
    pred_y = clf.predict(X_test)
    acc_clf, prec_clf, rec_clf, f1_clf, f2_clf = get_cm_metrics(y_test, pred_y, "validation set")
    return clf, scaler


def main():
    num_features = config.VectorizerConfig.NUM_FEATURES
    num_topics = config.LdaConfig.NUM_TOPICS
    num_iter = config.LdaConfig.NUM_ITERATIONS
    project_names = ['abinit', 'libmesh', 'lammps', 'mdanalysis']
    # project_names = ['mdanalysis']
    metrics = {}
    for p in project_names:
        raw_df = pre_process_text_dataframe(get_raw_df(p))
        test_raw_df = raw_df.sample(frac=0.2, replace=True)

        raw_train_df = raw_df[(raw_df['hash'].apply(lambda x: x not in test_raw_df['hash'].values))]

        vectorizer = get_vectorizer(num_features)
        X, feature_names = vectorize(raw_train_df['msg_str'], vectorizer)
        lda = fit_lda(X, num_topics, num_iter)
        processed_df = set_lda_topics_in_df(X, lda, num_topics, raw_train_df)
        final_df = transform_df(processed_df, lda, feature_names)
        clf, scaler = commitz_classifier(final_df)

        X_test_vector = vectorizer.transform(test_raw_df['msg_str'])
        processed_test_df = set_lda_topics_in_df(X_test_vector, lda, num_topics, test_raw_df)
        test_df = transform_df(processed_test_df, lda, feature_names)
        y_test_true = test_df['buggy']
        X_test_raw = test_df['word_prob'].values.tolist()
        X_test = scaler.transform(X_test_raw)
        y_test_pred = clf.predict(X_test)
        acc_clf, prec_clf, rec_clf, f1_clf, f2_clf = get_cm_metrics(y_test_pred, y_test_true, "Test Set")
        metrics[p] = [acc_clf, prec_clf, rec_clf, f1_clf, f2_clf]
    plot_metrices(metrics)
    print(metrics)


if __name__ == '__main__':
    main()
