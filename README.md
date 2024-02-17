# Fraud Detection using Computer Vision

## Table of Contents
1. Overview
2. Instructions
3. Dataset
4. Preprocessing
5. Model Development
6. Training
7. Inference
8. Challenges and Lessons Learned

## Overview

This project, titled "Fraud Detection using Computer Vision," is aimed at developing a robust and high-performance model utilizing computer vision techniques to classify images as either fraudulent or non-fraudulent within the context of insurance claims. The goal is to precisely identify fraudulent images, allowing insurance companies to evaluate the authenticity of a claim and make well-informed decisions regarding payouts.

## Author

**Pirate-Emperor**

## Instructions

1. **Dataset Preparation:** Ensure that the training set and test set files are available with the correct structure as described in the dataset section.

2. **Python Environment:** Set up a Python environment with the necessary libraries installed. You can use the provided code as a starting point.

3. **Adapt the Code:** Customize the code to fit your specific dataset and tweak the machine learning model as needed.

4. **Run the Model:** Train the model using the adapted code and make predictions on the test set.
    - Run all `.py` files named "Vxx.py" by modifying the `FOLD` variable in the `CFG` class.
    - Versions mentioned in `test.ipynb` without corresponding `Vxx.py` files are duplicated.
    - If two versions in the notebook have distinct names but only one file is included, run that file with `FOLD = 0` and `FOLD = 3`.

5. **Testing:**
    - Test the project using the `test.ipynb` notebook.
    - **EXTREMELY IMPORTANT:** Utilize folds from files such as `fold0_train.npy` for training indexes and `fold0_valid` for validation indexes.

6. **Evaluation:** Evaluate the model's performance using the macro F1 score. Consider further optimizations or model adjustments based on the evaluation results.

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

## Preprocessing
The initial step involved converting .jpg files to .npy files for faster loading. However, the presence of duplicated images did not significantly improve the performance of the model or the leaderboard score when removed.

## Model Development

The machine learning model is developed using Python and various libraries such as scikit-learn. The code incorporates strategies to handle class imbalance, crucial for achieving accurate predictions in fraud detection scenarios.
Various architectures were explored, including Efficientnet, Resnet, Resnext, Resnest, Seresnext, Vision Transformer, Swin, and EVA. The EfficientnetV2-XL model emerged as the most effective, achieving the highest score.

### Code Highlights

- **Class Imbalance Handling:** The code uses `compute_class_weight()` from scikit-learn to calculate class weights, addressing the imbalance in the dataset where 85% of images contain non-fraudulent claims.

- **Random Forest Classifier:** For demonstration purposes, a RandomForestClassifier is utilized. You can adapt the code to your specific dataset and model architecture.

- **Evaluation Metric:** The model is evaluated using the macro F1 score, as specified in the hackathon's requirements.

## Training
Four models were trained with slight variations in the base configuration. The base configuration included a learning rate of 2e-4, a CosineAnnealing learning rate scheduler, 10 epochs, and minimal weight decay. Augmentations such as Resize, HorizontalFlip, and VerticalFlip were used, along with BCEWithLogitsLoss. The last epoch was always chosen, as the cross-validation (CV) was not reliable.

## Model Specifications
The specifications of the four models were as follows:
- EfficientnetV2-XL, [512,512] image size
- EfficientnetV2-XL, [480,480] image size
- EfficientnetV2-XL, [480,480] image size, drop_rate=0.2, drop_path_rate=0.2, dropout_head=0.2
- EfficientnetV2-XL, [512,512] image size, drop_rate=0.2, drop_path_rate=0.2, dropout_head=0.2

## Inference
Using two folds from each model resulted in eight distinct predictions. Initially, the mean of predictions across all models was taken, but later, a voting ensemble was found to be more effective, providing a score boost of over +0.005. Test-Time Augmentation with Horizontal Flip TTA was also found to be beneficial. A threshold of 0.0575 was chosen to match the train distribution.

## Challenges and Lessons Learned
Several techniques, such as RandomBrightnessContrast, Noise, Blur, Cutout, and CoarseDropout, did not improve the score. Advanced optimizers like RangerLars and the new Prodigy optimizer did not outperform AdamW. Additionally, removing duplicate images did not yield better results, suggesting that a different approach to removing duplicates may be necessary for a stable CV.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to reach out to the author, Pirate-Emperor, for any questions or clarifications. Happy coding!