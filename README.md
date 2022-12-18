# Principle Component Analysis Wine Dataset


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Principal Component Analysis, or PCA for short, is a way of taking a bunch of data and finding patterns in it. Imagine you have a bunch of points on a graph, and you want to know if there's a way to draw a line that goes through as many of the points as possible. PCA can help you find that line!

Here's how it works: First, you find the line that goes through the points and has the most points on it. This line is called the first principal component. Then, you find a line that goes through the points and has the second most points on it. This line is called the second principal component. You can keep going and find more principal components if you want, but usually two or three are enough.

PCA is useful because it can help you understand the patterns in your data. For example, if you have a bunch of data about people's heights and weights, PCA can help you find out if there's a pattern to how tall and heavy people are. You might find that there's a line that goes through most of the points, which means that people's heights and weights are related in some way.
## Uses

- Dimensionality reduction: PCA can be used to reduce the number of dimensions in a dataset, which can make it easier to visualize and analyze.
- Data visualization: PCA can be used to create scatterplots and other visualizations that can help you understand patterns in your data.
- Feature selection: PCA can be used to identify the most important features in a dataset, which can be useful for building predictive models.
- Data compression: PCA can be used to compress large datasets, which can make it faster and more efficient to store and analyze the data.
- Noise reduction: PCA can be used to remove noise and outliers from a dataset, which can improve the accuracy of your analysis.


## Steps to implement PCA on wine dataset.

- Start by importing the necessary libraries, including scikit-learn and numpy. You'll also need to import the wine dataset from scikit-learn:
  ```
  import numpy as np
  from sklearn import datasets
  # Load the wine dataset
  wine = datasets.load_wine()
  ```
- Next, split the dataset into training and test sets. You can use scikit-learn's train_test_split function to do this:
   ```
    from sklearn.model_selection import train_test_split
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)
   ``` 

- Now, you'll need to standardize the data. PCA is sensitive to the scale of the features, so it's important to make sure that all the features are on the same scale. You can use scikit-learn's StandardScaler to do this:
   ```
    from sklearn.preprocessing import StandardScaler

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
   ``` 

- Now, you're ready to apply PCA to the data. First, import the PCA class from scikit-learn and create an instance of it. You'll also need to specify the number of components you want to keep:
   ```
    from sklearn.decomposition import PCA
    # Create a PCA instance with 2 components
    pca = PCA(n_components=2)
   ``` 

- Now, you can fit the PCA model to the training data and transform it:
   ```
    # Fit the model to the training data and transform it
    X_train_pca = pca.fit_transform(X_train)
   ``` 

- You can also transform the test data using the fitted PCA model:
   ```
   # Transform the test data using the fitted model
   X_test_pca = pca.transform(X_test)
   ``` 

- Now, you can use the transformed data to build a predictive model. For example, you could use a support vector machine (SVM) to classify the wine types:
   ```
          from sklearn.svm import SVC

          # Create an SVC model
          model = SVC()

          # Fit the model to the training data
          model.fit(X_train_pca, y_train)

          # Evaluate the model on the test data
          print(model.score(X_test_pca, y_test))
   ``` 

## References
- https://www.youtube.com/watch?v=oiusrJ0btwA&t=552s
- https://arxiv.org/abs/1404.1100
- https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476590
- https://machinelearningmastery.com/gentle-introduction-principal-component-analysis/
   
