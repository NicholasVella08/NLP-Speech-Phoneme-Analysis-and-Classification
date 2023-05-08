import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def read_csv_file(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file)
        formants = []
        IDClass = []
        for i, row in enumerate(csv_reader):
            # Ignore the first row as it contains column names.
            if i == 0:
                continue
            # Extract the Class ID and F1,F2,F3 values from each row.
            classID = int(row[4])
            f1 = float(row[5])
            f2 = float(row[6])
            f3 = float(row[7])
            # Append new entry to the lists.
            formants.append([f1, f2, f3])
            IDClass.append(classID)
    return formants, IDClass



def build_confusion_matrix(X, Y):
    print("\nThe confusion matrix is:")
    print(confusion_matrix(X, Y, labels=[1, 2, 3]))
    # Micro - calculates by counting total true positives, false negatives and false positives.
    print("F1 score:" + str(round(f1_score(X, Y, average='micro'), 4)))


def perform_knn_classification(X, y, split_percent, rand_seed, num_neighbors, distance_metric):
    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percent, random_state=rand_seed)
    # Create a KNN classifier with the specified number of neighbors and distance metric.
    knn_classifier = KNeighborsClassifier(n_neighbors=num_neighbors, metric=distance_metric)
    # Train the model by passing in our training data to fit our model to the training data.
    knn_classifier.fit(X_train, y_train)
    # Make predictions on the test data.
    y_pred = knn_classifier.predict(X_test)
    # Build and display the confusion matrix and F1 score based on predicted and actual labels.
    build_confusion_matrix(y_test, y_pred)




# Read the data from the csv file.
#formants, IDClass = read_csv_file("Feature Extraction.csv")
formants, IDClass = read_csv_file("male.csv")
#print(formants)
#print(IDClass)
# Set customizable values for the KNN classifier and splitting percentage.
split_percentage = 0.25
num_neighbors = 5
distance_metric = "manhattan"
# Perform KNN classification 5 times for different training/testing splits by changing the random seed value.
for i in range(5):
    perform_knn_classification(formants, IDClass, split_percentage, i, num_neighbors, distance_metric)



