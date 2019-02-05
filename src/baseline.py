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

from src.config import Metrics

corrective_seed = {"fix", "bug", "problem", "incorrect", "correct", "error", "fixup", "fail"}
adaptive_seed = {"new", "change", "patch", "add", "modify", "update"}
perfective_seed = {"style", "move", "removal", "cleanup", "unneeded", "rework"}

mauckza_corrective_seed = {'active', 'against', 'already', 'bad', 'block', 'bug', 'build', 'call', 'case', 'catch',
                           'cause', 'character', 'compile', 'correctly', 'create', 'different', 'dump', 'error',
                           'except', 'exist', 'explicitly', 'fail', 'failure', 'fast', 'fix', 'format', 'good', 'hack',
                           'hard', 'help', 'init', 'instead', 'introduce', 'issue', 'lock', 'log', 'logic', 'look',
                           'merge', 'miss', 'null', 'oops', 'operation', 'operations', 'pass', 'previous', 'previously',
                           'probably', 'problem', 'properly', 'random', 'recent', 'request', 'reset', 'review', 'run',
                           'safe', 'set', 'similar', 'simplify', 'special', 'test', 'think', 'try', 'turn', 'valid',
                           'wait', 'warn', 'warning', 'wrong'}
mauckza_adaptive_seed = {'active', 'add', 'additional', 'against', 'already', 'appropriate', 'available', 'bad',
                         'behavior', 'block', 'build', 'call', 'case', 'catch', 'change', 'character', 'compatibility',
                         'compile', 'config', 'configuration', 'context', 'correctly', 'create', 'currently', 'default',
                         'different', 'documentation', 'dump', 'easier', 'except', 'exist', 'explicitly', 'fail',
                         'fast', 'feature', 'format', 'future', 'good', 'hack', 'hard', 'header', 'help', 'include',
                         'information', 'init', 'inline', 'install', 'instead', 'internal', 'introduce', 'issue',
                         'lock', 'log', 'logic', 'look', 'merge', 'method', 'necessary', 'new ', 'old', 'operation',
                         'operations', 'pass', 'patch', 'previous', 'previously', 'probably', 'properly',
                         'protocol provide', 'random', 'recent', 'release', 'replace ,request', 'require', 'reset',
                         'review', 'run', 'safe', 'security', 'set', 'similar', 'simple', 'simplify', 'special',
                         'structure', 'switch', 'test', 'text', 'think', 'trunk', 'try', 'turn', 'useful', 'user',
                         'valid', 'version', 'wait'}
mauckza_perfective_seed = {'cleanup', 'consistent', 'declaration', 'definition', 'header', 'include', 'inline', 'move',
                           'prototype', 'removal', 'static', 'style', 'unused', 'variable', 'warning', 'whitespace'}
mauckza_blacklist_seed = {'cvs2svn', 'cvs', 'svn'}

lemmatizer = WordNetLemmatizer()
rt = RegexpTokenizer(r'[^\W_]+|[^\W_\s]+')
# stopset = set(stopwords.words('english'))
stopset = {"the"}

data_path = "/Users/saurabh/Downloads/ncsu/study/thesis/project/data/"

mapper = lambda word_list: 0 if any(word in corrective_seed for word in word_list) else 1 if any(
    word in adaptive_seed for word in word_list) else 2 if any(
    word in perfective_seed for word in word_list) else -1

mauckza_mapper = lambda word_list: 0 if any(word in mauckza_corrective_seed for word in word_list) else 1 if any(
    word in mauckza_adaptive_seed for word in word_list) else 2 if any(
    word in mauckza_perfective_seed for word in word_list) else 3 if any(
    word in mauckza_blacklist_seed for word in word_list) else -1


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


def baseline(df):
    df["mockus_class"] = df["msg"].apply(mapper)
    df["mauckza_class"] = df["msg"].apply(mauckza_mapper)
    df["mockus"] = df["mockus_class"].apply(lambda c: 1 if c == 0 else 0)
    df["mauckza"] = df["mauckza_class"].apply(lambda c: 1 if c == 0 else 0)
    print(confusion_matrix(df["buggy"], df["mockus"]))
    get_cm_metrics(df["buggy"], df["mockus"], "mockus")
    print(confusion_matrix(df["buggy"], df["mauckza"]))
    get_cm_metrics(df["buggy"], df["mauckza"], "mauckza")


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


def main():
    project_names = [['abinit'], ['libmesh'], ['mdanalysis']]
    # project_names = [['abinit']]
    for idx, p in enumerate(project_names):
        print("Playing with {}".format(p))
        raw_df = pre_process_text_dataframe(get_raw_df(p))
        train_df, test_df = train_test_split(raw_df, test_size=0.2)
        baseline(train_df)


if __name__ == '__main__':
    main()
