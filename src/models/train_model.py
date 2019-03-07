import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys
sys.path.append("/Users/oscarwyatt/data_science/pageview_predictor/src/")
import utils
from sklearn.model_selection import KFold
import traceback
import warnings
import sys
from sklearn.ensemble import ExtraTreesClassifier
import pickle
import math

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

def draw_content_item_distribution_graph(pageview_values, descretizer, view_numbers):
    binned_page_views = discretizer.transform(view_numbers)
    count = {}
    for bin in binned_page_views:
        number = count.get(bin[0], 0)
        count[bin[0]] = number + 1

    plt.plot(binned_page_views, view_numbers, f'r--')
    params = discretizer.get_params()
    plt.title("Distribution of binned regulation content items log10 page views with " + str(params['n_bins']) + " bins and " + params[
        'strategy'] + " strategy")
    plt.ylabel('Log10 of page views')
    plt.xlabel("Bin edges at: " + ", ".join([str(x) for x in discretizer.bin_edges_.tolist()[0]]) + "\n" +  "# items in each bin" + str(count))
    plt.show()


def show_feature_importance(X_extra_features, y):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    forest.fit(X_extra_features, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_extra_features.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_extra_features.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_extra_features.shape[1]), indices)
    plt.xlim([-1, X_extra_features.shape[1]])
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        default="vectorize", help="Choose between 'vectorize' and 'doc_to_vec'")
    parser.add_argument("-d", "--distribution-graph",
                        default=False, help="Draw a graph of the distribution of the content item pageviews")
    parser.add_argument("-i", "--feature-importance", default=False, action='store_true', help="Show graph of the importance of various features")
    parser.add_argument("-c", "--confusion-matrix", default=False, action='store_true', help="Show confusion matrices during training")
    parser.add_argument("-k", "--k-fold", default=False, action='store_true', help="Perform K-Fold")
    parser.add_argument("-t", "--test", default=False, action='store_true', help="Split into training and test")
    args = vars(parser.parse_args())

    model_file = "vectorize"
    if args["model"] != "vectorize":
        model_file = "doc_to_vec"
    with open("data/processed/" + model_file + "_X", 'rb') as fp:
        X = pickle.load(fp)
    with open("data/processed/" + model_file + "_y", 'rb') as fp:
        y = np.asarray(pickle.load(fp))

    if args["distribution_graph"]:
        pageviews = utils.load_pageviews()
        discretizer, view_numbers = utils.generate_discretizer(pageviews)
        draw_content_item_distribution_graph(pageviews, discretizer, view_numbers)

    if args["feature_importance"]:
        show_feature_importance(X, y)

    if args["k_fold"]:
        kf = KFold(n_splits=5)
        kf.get_n_splits(X)

        f1_scores = []
        accuracy_scores = []
        confusion_matrix = np.zeros((utils.number_bins(),utils.number_bins()))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            f1_score, accuracy_score, fold_confusion_matrix = utils.train_and_test_logistic_regression(X_train, y_train, X_test, y_test, args["confusion_matrix"])
            confusion_matrix = confusion_matrix + fold_confusion_matrix
            f1_scores.append(f1_score)
            accuracy_scores.append(accuracy_score)

        if args["confusion_matrix"]:
            utils.plot_confusion_matrix(confusion_matrix)

        print("Average f1 score :" + str(np.mean(f1_scores)))
        print("Average accuracy score :" + str(np.mean(accuracy_scores)))

    if args["test"]:
        count = len(X)
        split_index = math.floor(count * 0.8)
        X_train = X[0:split_index]
        X_test = X[split_index:]
        y_train = y[0:split_index]
        y_test = y[split_index:]
        f1_score, accuracy_score, confusion_matrix = utils.train_and_test_logistic_regression(X_train, y_train, X_test, y_test, args["confusion_matrix"])
        confusion_matrix

        if args["confusion_matrix"]:
            utils.plot_confusion_matrix(confusion_matrix)

        print("Accuracy score for test data :" + str(accuracy_score))
        print("F1 score for test data :" + str(f1_score))


    model = utils.train_logistic_regression(X, y)
    model_filename = "logistic_regression_model.pkl"
    pickle.dump(model, open(model_filename, 'wb'))
    print("Model saved to " + model_filename)

if __name__== "__main__":
    main()
