# Alphabet Soup Charity Classifier

## Report on the Neural Network Model for Alphabet Soup

### Overview of the Analysis:

The purpose of this analysis is to develop a deep learning model using a neural network to predict the success of funding applications submitted to Alphabet Soup, a charitable organization. By training a model on historical data regarding past funding applications, Alphabet Soup aims to streamline its approval process and allocate resources more effectively to maximize impact.

### Results:

#### Data Preprocessing:

- **Target Variable:** The target variable for the model is the "IS_SUCCESSFUL" column, indicating whether a funding application was successful (1) or not (0).
- **Features:** Features include various columns providing information about each funding application, such as "APPLICATION_TYPE", "AFFILIATION", "CLASSIFICATION", etc.
- **Variables to Remove:** Columns like "EIN" (Employer Identification Number), "NAME" (Organization Name), and "SPECIAL_CONSIDERATIONS" were removed as they are neither targets nor features. Additionally, "ASK_AMT" was removed due to its lack of predictive value.

#### Compiling, Training, and Evaluating the Model:

- **Model Architecture:** The neural network model consists of three hidden layers with 200, 100, and 50 neurons, respectively. Activation functions used are Leaky ReLU, ELU, and tanh to introduce non-linearity.
- **Achievement of Target Model Performance:** The model achieved an accuracy of approximately 72.61%, falling short of the target performance threshold of 75%.

### Optimization Attempts:

Several optimization attempts were made to enhance the model's performance:

- **Data Preprocessing:** Techniques such as binning rare occurrences and dropping irrelevant columns were employed to improve data quality and reduce noise.
- **Model Architecture:** The model architecture was adjusted by increasing the number of neurons, adding more hidden layers, and using different activation functions to capture complex patterns in the data.
- **Training Parameters:** Learning rate and the number of epochs were fine-tuned to optimize model convergence and prevent overfitting.

### Summary:

Despite optimization attempts, the deep learning model for Alphabet Soup achieved a moderate accuracy of 72.61%, falling short of the target threshold. Future optimization strategies could include further feature engineering, hyperparameter tuning, exploring ensemble methods, and addressing data imbalance. By implementing these strategies, it is possible to enhance the model's performance and achieve the desired accuracy for predicting funding application success for Alphabet Soup.


## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

## Instructions

### Step 1: Preprocess the Data

Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
   - What variable(s) are the target(s) for your model?
   - What variable(s) are the feature(s) for your model?
   
2. Drop the EIN and NAME columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

6. Use pd.get_dummies() to encode categorical variables.

7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. If necessary, add a second hidden layer with an appropriate activation function.

5. Create an output layer with an appropriate activation function.

6. Check the structure of the model.

7. Compile and train the model.

8. Create a callback that saves the model's weights every five epochs.

9. Evaluate the model using the test data to determine the loss and accuracy.

10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

### Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    - Dropping more or fewer columns.
    - Creating more bins for rare occurrences in columns.
    - Increasing or decreasing the number of values for each bin.
    - Add more neurons to a hidden layer.
    - Add more hidden layers.
    - Use different activation functions for the hidden layers.
    - Add or reduce the number of epochs to the training regimen.

Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

## References

IRS. Tax Exempt Organization Search Bulk Data Downloads. [https://www.irs.gov/](https://www.irs.gov/)