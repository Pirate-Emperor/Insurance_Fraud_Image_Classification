# Fraud Detection using Computer Vision

## Overview

This project, titled "Fraud Detection using Computer Vision," is aimed at developing a robust and high-performance model utilizing computer vision techniques to classify images as either fraudulent or non-fraudulent within the context of insurance claims. The goal is to precisely identify fraudulent images, allowing insurance companies to evaluate the authenticity of a claim and make well-informed decisions regarding payouts.

## Author

**Pirate-Emperor**

## Dataset

### Training Set

The training set consists of two files:
- **images folder:** Contains a diverse dataset of car images used for training the model.
- **train.csv:** Contains the labels of each image present in the training set.

#### Data Description

- **image_id:** Unique identifier of the row.
- **filename:** Name of the image.
- **label:** Binary variable containing 0 or 1. 0 indicates a non-fraudulent claim, and 1 indicates a fraudulent claim.

### Test Set

The test set also consists of two files:
- **images folder:** Contains the test images for which predictions are to be made.
- **test.csv:** Contains the unique identifiers of each image present in the test set.

#### Data Description

- **image_id:** Unique identifier of the row.
- **filename:** Name of the image.

### Sample Submission

The sample submission file contains two columns:
- **image_id:** Unique identifier of an image.
- **label:** Binary variable containing 0 or 1. 0 indicates a non-fraudulent claim, and 1 indicates a fraudulent claim.

## Model Development

The machine learning model is developed using Python and various libraries such as scikit-learn. The code incorporates strategies to handle class imbalance, crucial for achieving accurate predictions in fraud detection scenarios.

### Code Highlights

- **Class Imbalance Handling:** The code uses `compute_class_weight()` from scikit-learn to calculate class weights, addressing the imbalance in the dataset where 85% of images contain non-fraudulent claims.

- **Random Forest Classifier:** For demonstration purposes, a RandomForestClassifier is utilized. You can adapt the code to your specific dataset and model architecture.

- **Evaluation Metric:** The model is evaluated using the macro F1 score, as specified in the hackathon's requirements.

## Instructions

1. **Dataset Preparation:** Ensure that the training set and test set files are available with the correct structure as described in the dataset section.

2. **Python Environment:** Set up a Python environment with the necessary libraries installed. You can use the provided code as a starting point.

3. **Adapt the Code:** Customize the code to fit your specific dataset and tweak the machine learning model as needed.

4. **Run the Model:** Train the model using the adapted code and make predictions on the test set.

5. **Evaluation:** Evaluate the model's performance using the macro F1 score. Consider further optimizations or model adjustments based on the evaluation results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to reach out to the author, Pirate-Emperor, for any questions or clarifications. Happy coding!