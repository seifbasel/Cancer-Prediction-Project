# Cancer Prediction Project

## Overview
This project aims to predict the likelihood of cancer based on various features, including smoking habits and other relevant factors. We employed three different machine learning models for classification: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Decision Tree.

## Table of Contents
- [Dataset](#dataset)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset used in this project is stored in the `cancer_prediction_dataset.csv` file. It contains information about individuals, including features like age, smoking habits, and the target variable indicating whether the individual has cancer (1) or not (0).

## Models
1. **K-Nearest Neighbors (KNN):** Utilized a KNN classifier with three neighbors for prediction.
2. **Support Vector Machine (SVM):** Used a linear SVM classifier with a regularization parameter of 1.
3. **Decision Trees:** Employed both unpruned and pruned Decision Tree classifiers for comparison.

## Usage
To run the project locally, follow these steps:
1. Clone the repository: `git clone https://github.com/your-username/cancer-prediction.git`
2. Navigate to the project directory: `cd cancer-prediction`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the main script: `python main.py`

## Results
- KNN Accuracy: [Your KNN Accuracy]
- SVM Accuracy: [Your SVM Accuracy]
- Decision Tree (Unpruned) Accuracy: [Your Decision Tree (Unpruned) Accuracy]
- Decision Tree (Pruned) Accuracy: [Your Decision Tree (Pruned) Accuracy]

## Visualizations
- Confusion Matrix for KNN
- Confusion Matrix for SVM
- Confusion Matrix for Decision Tree (Unpruned)
- Confusion Matrix for Decision Tree (Pruned)
- ROC Curve for SVM
- Decision Tree (Unpruned) Visualization
- Decision Tree (Pruned) Visualization

## Contributing
Contributions are welcome! If you have any ideas for improvement or bug fixes, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
