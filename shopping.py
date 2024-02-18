import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    with open(filename, newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        for row in reader:
            # Convert values to the required types
            admin, admin_duration, info, info_duration, product, product_duration, \
            bounce_rates, exit_rates, page_values, special_day, month, \
            os, browser, region, traffic_type, visitor_type, weekend, revenue = row

            evidence.append([
                int(admin),
                float(admin_duration),
                int(info),
                float(info_duration),
                int(product),
                float(product_duration),
                float(bounce_rates),
                float(exit_rates),
                float(page_values),
                float(special_day),
                ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(month),
                int(os),
                int(browser),
                int(region),
                int(traffic_type),
                1 if visitor_type == "Returning_Visitor" else 0,
                1 if weekend == "TRUE" else 0
            ])

            labels.append(1 if revenue == "TRUE" else 0)

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positives = sum((actual == 1) and (predicted == 1) for actual, predicted in zip(labels, predictions))
    true_negatives = sum((actual == 0) and (predicted == 0) for actual, predicted in zip(labels, predictions))

    sensitivity = true_positives / labels.count(1)
    specificity = true_negatives / labels.count(0)

    return sensitivity, specificity


if __name__ == "__main__":
    main()


# import csv
# import sys

# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier

# TEST_SIZE = 0.4


# def main():

#     # Check command-line arguments
#     if len(sys.argv) != 2:
#         sys.exit("Usage: python shopping.py data")

#     # Load data from spreadsheet and split into train and test sets
#     evidence, labels = load_data(sys.argv[1])
#     X_train, X_test, y_train, y_test = train_test_split(
#         evidence, labels, test_size=TEST_SIZE
#     )

#     # Train model and make predictions
#     model = train_model(X_train, y_train)
#     predictions = model.predict(X_test)
#     sensitivity, specificity = evaluate(y_test, predictions)

#     # Print results
#     print(f"Correct: {(y_test == predictions).sum()}")
#     print(f"Incorrect: {(y_test != predictions).sum()}")
#     print(f"True Positive Rate: {100 * sensitivity:.2f}%")
#     print(f"True Negative Rate: {100 * specificity:.2f}%")


# def load_data(filename):
#     """
#     Load shopping data from a CSV file `filename` and convert into a list of
#     evidence lists and a list of labels. Return a tuple (evidence, labels).

#     evidence should be a list of lists, where each list contains the
#     following values, in order:
#         - Administrative, an integer
#         - Administrative_Duration, a floating point number
#         - Informational, an integer
#         - Informational_Duration, a floating point number
#         - ProductRelated, an integer
#         - ProductRelated_Duration, a floating point number
#         - BounceRates, a floating point number
#         - ExitRates, a floating point number
#         - PageValues, a floating point number
#         - SpecialDay, a floating point number
#         - Month, an index from 0 (January) to 11 (December)
#         - OperatingSystems, an integer
#         - Browser, an integer
#         - Region, an integer
#         - TrafficType, an integer
#         - VisitorType, an integer 0 (not returning) or 1 (returning)
#         - Weekend, an integer 0 (if false) or 1 (if true)

#     labels should be the corresponding list of labels, where each label
#     is 1 if Revenue is true, and 0 otherwise.
#     """
#     raise NotImplementedError


# def train_model(evidence, labels):
#     """
#     Given a list of evidence lists and a list of labels, return a
#     fitted k-nearest neighbor model (k=1) trained on the data.
#     """
#     raise NotImplementedError


# def evaluate(labels, predictions):
#     """
#     Given a list of actual labels and a list of predicted labels,
#     return a tuple (sensitivity, specificity).

#     Assume each label is either a 1 (positive) or 0 (negative).

#     `sensitivity` should be a floating-point value from 0 to 1
#     representing the "true positive rate": the proportion of
#     actual positive labels that were accurately identified.

#     `specificity` should be a floating-point value from 0 to 1
#     representing the "true negative rate": the proportion of
#     actual negative labels that were accurately identified.
#     """
#     raise NotImplementedError


# if __name__ == "__main__":
#     main()
