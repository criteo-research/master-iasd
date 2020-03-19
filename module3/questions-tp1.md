* Write a python function to split the rows of the file randomly in a train and test files
* Read the file in a incremental manner, preprocess into feature and label batches, and use SGDClassifier with partial_fit to fit the model incrementally
* Evaluate the resulting model (log loss and AUC) on the test file
* Write a function that converts the train and test files into Vowpal Wabbit's format https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format
* Train a vowpal wabbit model either through command line or through python
* Generate predictions using the resulting vw model on the test file
* Compute the AUC and log loss of the resulting predictions