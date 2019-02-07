"""
Andreas Mauczka - dictionary based algorithm
Tracing Your Maintenance Work – A Cross-Project Validation of an Automated Classification Dictionary for Commit Messages
https://dl.acm.org/citation.cfm?id=2259307

1. Classify the commit using the “seed” dictionary
2. If the total percentage of classified commits is greater than 80%, EXIT
3. Count the appearances of all words in the commit messages of the nonclassified
commits and order them by frequency
4. Choose a set of words from the top of the list and add these as a test set to
the existing dictionary
5. Count the number of appearances of every word in the test set in each
category
6. If the number of appearances of a word in a category is at least 1.5 times
of the appearances of the same word in the other categories, add it to the
dictionary with a weight of 2 and remove it from the test set
7. If the number of appearances of a word in two classes is at least 1.5 times of
the appearances of the same word in the third class, add it to the dictionary
to both classes with a weight of 1 and remove it from the test set
8. If neither 6 or 7 are true, remove the word from the test set and do not add
it to the dictionary
9. Go to Step 2


Corrective: active, against, already, bad, block, bug, build, call, case, catch,
cause(2), character, compile, correctly, create, different, dump, error(2), except,
exist, explicitly, fail, failure(2), fast, fix(2), format, good, hack, hard,
help, init, instead, introduce, issue, lock, log, logic, look, merge, miss(2),
null(2), oops(2), operation, operations, pass, previous, previously, probably,
problem, properly, random, recent, request, reset, review, run, safe, set, similar,
simplify, special, test, think, try, turn, valid, wait, warn(2), warning,
wrong(2)

Adaptive: active, add(2), additional(2), against, already, appropriate(2),
available(2), bad, behavior, block, build, call, case, catch, change(2), character,
compatibility(2), compile, config(2), configuration(2), context(2), correctly,
create, currently(2), default(2), different, documentation(2), dump,
easier(2), except, exist, explicitly, fail, fast, feature(2), format, future(2),
good, hack, hard, header, help, include, information(2), init, inline, install(2),
instead, internal(2), introduce, issue, lock, log, logic, look, merge, method(2),
necessary(2), new (2), old(2), operation, operations, pass, patch(2), previous,
previously, probably, properly, protocol(2) provide(2), random, recent,
release(2), replace(2) ,request, require(2), reset, review, run, safe, security(2),
set, similar, simple(2), simplify, special, structure(2), switch(2), test, text(2),
think, trunk(2), try, turn, useful(2), user(2), valid, version(2), wait

Perfective: cleanup(2), consistent(2), declaration(2), definition(2), header, include,
inline, move(2), prototype(2), removal(2), static(2), style(2), unused(2),
variable(2), warning, whitespace(2)

Blacklist: cvs2svn, cvs, svn


"""

import glob
import os
import re
import string

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from config import Metrics

corrective_seed = {"fix", "bug", "problem", "incorrect", "correct", "error", "fixup", "fail"}
adaptive_seed = {"new", "change", "patch", "add", "modify", "update"}
perfective_seed = {"style", "move", "removal", "cleanup", "unneeded", "rework"}

mauckza_blacklist_seed = {'cvs2svn', 'cvs', 'svn'}

mauckza_corrective_seed = {'bugfix', 'bug', 'cause', 'error', 'failure', 'fix', 'miss', 'null', 'warn', 'wrong', 'bad',
                           'correct', 'incorrect', 'problem', 'opps', 'valid', 'invalid', 'fail', 'bad', 'dump',
                           'except'}
mauckza_adaptive_seed = {'add', 'new', 'create', 'feature', 'function', 'appropriate', 'available', 'change',
                         'compatibility', 'config', 'configuration', 'text', 'current', 'default', 'easier', 'future',
                         'information', 'internal', 'method', 'necessary', 'old', 'patch', 'protocol', 'provide',
                         'release', 'replace', 'require', 'security', 'simple', 'structure', 'switch', 'context',
                         'trunk', 'useful', 'user', 'version', 'install', 'introduce', 'faster', 'init'}
mauckza_perfective_seed = {'clean', 'cleanup', 'consistent', 'declaration', 'definition', 'documentation', 'move',
                           'prototype', 'remove', 'static', 'style', 'unused', 'variable', 'whitespace', 'header',
                           'include', 'dead', 'inefficient', 'useless'}

mapper = lambda word_list: 0 if any(word in corrective_seed for word in word_list) else 1 if any(
    word in adaptive_seed for word in word_list) else 2 if any(
    word in perfective_seed for word in word_list) else -1

mauckza_mapper = lambda word_list: 0 if any(word in mauckza_corrective_seed for word in word_list) else 1 if any(
    word in mauckza_adaptive_seed for word in word_list) else 2 if any(
    word in mauckza_perfective_seed for word in word_list) else 3 if any(
    word in mauckza_blacklist_seed for word in word_list) else -1

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


def baseline(df, mockus=False):
    if mockus:
        df["mockus_class"] = df["msg"].apply(mapper)
        df["mockus"] = df["mockus_class"].apply(lambda c: 1 if c == 0 else 0)
        return get_cm_metrics(df["buggy"], df["mockus"], "mockus")
    else:
        df["mauckza_class"] = df["msg"].apply(mauckza_mapper)
        df["mauckza"] = df["mauckza_class"].apply(lambda c: 1 if c == 0 else 0)
        return get_cm_metrics(df["buggy"], df["mauckza"], "mauckza")


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
    return raw_df


def main(test_mockus=False):
    project_names = [['abinit'], ['libmesh'], ['mdanalysis']]
    # project_names = [['abinit']]
    metrics = {}
    for idx, p in enumerate(project_names):
        project_key = '+'.join(p)
        print("Playing with {}".format(project_key))
        raw_df = pre_process_text_dataframe(get_raw_df(p))
        train_df, test_df = train_test_split(raw_df, test_size=0.01)
        metrics[project_key] = baseline(train_df, mockus=test_mockus)
    return metrics


if __name__ == '__main__':
    test_mockus = True
    f_name = "mockus.csv" if test_mockus else "mauczka.csv"
    final_mean_metrics = {}
    sum_metrics = {}
    for i in range(20):
        metrics_map = main(test_mockus=test_mockus)
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
