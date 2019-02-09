from nose.tools import eq_, assert_almost_equals, assert_greater_equal
from snlp import preproc, clf_base, constants, hand_weights, evaluation, naive_bayes, logreg
import numpy as np



def setup_module():
    global x_tr, y_tr, x_dv, y_dv, counts_tr, x_dv_pruned, x_tr_pruned
    global labels
    global vocab
    global X_tr, X_tr_var, X_dv_var, Y_tr, Y_dv, Y_tr_var, Y_dv_var

    y_tr,x_tr = preproc.read_data('data/rock-lyrics-train.csv',preprocessor=preproc.bag_of_words)
    labels = set(y_tr)

    counts_tr = preproc.aggregate_counts(x_tr)

    y_dv,x_dv = preproc.read_data('data/rock-lyrics-dev.csv',preprocessor=preproc.bag_of_words)

    x_tr_pruned, vocab = preproc.prune_vocabulary(counts_tr, x_tr, 10)
    x_dv_pruned, _ = preproc.prune_vocabulary(counts_tr, x_dv, 10)

    X_tr = preproc.make_numpy(x_tr_pruned,vocab)
    X_dv = preproc.make_numpy(x_dv_pruned,vocab)
    label_set = sorted(list(set(y_tr)))
    Y_tr = np.array([label_set.index(y_i) for y_i in y_tr])
    Y_dv = np.array([label_set.index(y_i) for y_i in y_dv])


def test_d4_1_numpy():
    global x_dv, counts_tr
    
    x_dv_pruned, vocab = preproc.prune_vocabulary(counts_tr,x_dv,10)
    X_dv = preproc.make_numpy(x_dv_pruned,vocab)
    eq_(X_dv.sum(), 113386)
    eq_(X_dv.sum(axis=1)[4], 95)
    eq_(X_dv.sum(axis=1)[144], 154)

    eq_(X_dv.sum(axis=0)[10], 4)


def test_d4_2_convert():
    global x_dv, counts_tr

    y_tr_conv = preproc.convert_categ_label_to_vec(y_tr)
    eq_(len(y_tr_conv[0]), 4)
    print(y_tr_conv[10])
    eq_(y_tr_conv[10][1], 0)
    eq_(y_tr_conv[10][3], 1)
    eq_(y_tr_conv[10][2], 0)



def test_d4_3_logreg():
    global X_tr, Y_tr, X_dv_var

    model = logreg.build_linear(X_tr,Y_tr)
    scores = model.forward(X_dv_var)
    eq_(scores.size()[0], 450)
    eq_(scores.size()[1], 4)


