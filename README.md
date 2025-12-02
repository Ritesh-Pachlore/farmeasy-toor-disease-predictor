# Crop Disease Prediction System

A machine learning-based web application for predicting diseases in Toor (Pigeon Pea) crops based on growth stage and sowing date.

## Features

- Predict crop diseases using machine learning models
- Web interface built with Flask
- Supports Toor (Pigeon Pea) crop
- Determines growth stage automatically based on days since sowing
- Provides prediction confidence scores

## Dataset

The system uses a dataset (`dataset/toor_crop_dataset.csv`) containing:

- Crop type
- Growth stage
- Sowing date
- Disease labels

## Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- XGBoost
- LightGBM
- CatBoost

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/crop-disease-prediction.git
   cd crop-disease-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (optional, model.pkl is already included):

   ```bash
   python train.py
   ```

4. Run the application:

   ```bash
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000`

## Usage

1. Select "Toor (Pigeon Pea)" as the crop
2. Enter the sowing date in YYYY-MM-DD format
3. Click "Predict Disease"
4. View the prediction results including:
   - Predicted disease
   - Current growth stage
   - Days since sowing
   - Prediction confidence

## Model Training

The system trains multiple machine learning models and selects the best performing one:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- LightGBM
- CatBoost
- SVM
- KNN

## Project Structure

```
crop-disease-prediction/
├── app.py                 # Flask web application
├── train.py               # Model training script
├── model.pkl              # Trained model (generated)
├── dataset/
│   └── toor_crop_dataset.csv
├── templates/
│   └── index.html         # Web interface
├── catboost_info/         # Training logs (generated)
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Deployment on Render

### Prerequisites

1. Upload your project to GitHub (see instructions below)
2. Sign up for a free account at [Render.com](https://render.com)

### Step 1: Upload to GitHub

1. Create a new repository on GitHub
2. Push your code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Crop disease prediction system"
   git remote add origin https://github.com/your-username/your-repo-name.git
   git push -u origin master
   ```

### Step 2: Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: crop-disease-prediction (or your choice)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
5. Add environment variables (optional):
   - `FLASK_SECRET`: A random secret key for production
   - `FLASK_DEBUG`: 0 (for production)
   - `PORT`: Will be set automatically by Render
6. Click "Create Web Service"

### Troubleshooting Render Deployment

If you encounter the error "Cannot import 'setuptools.build_meta'":

- This is a common issue with pip/setuptools compatibility on Python 3.13
- Try updating the **Build Command** in Render to: `pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`
- Alternatively, you can try changing the Python version in Render to 3.11 or 3.12 if available
- If using Python 3.13, ensure setuptools version is >= 68.0.0

**Alternative Build Command:**

```
pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
```

### Step 3: Access Your App

Once deployed, Render will provide a URL like `https://your-app-name.onrender.com`

## Uploading to GitHub (Alternative)

If you haven't uploaded to GitHub yet:

### Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right and select "New repository"
3. Name your repository (e.g., "crop-disease-prediction")
4. Add a description
5. Choose public or private
6. Do NOT initialize with README, .gitignore, or license (we already have them)
7. Click "Create repository"

### Step 2: Initialize Git and Push to GitHub

1. Open terminal/command prompt in your project directory
2. Initialize Git repository:
   ```bash
   git init
   ```
3. Add all files:
   ```bash
   git add .
   ```
4. Commit the files:
   ```bash
   git commit -m "Initial commit: Crop disease prediction system"
   ```
5. Add the remote repository:

   ```bash
   git remote add origin https://github.com/your-username/your-repo-name.git
   ```

   Replace `your-username` and `your-repo-name` with your actual GitHub username and repository name.

6. Push to GitHub:
   ```bash
   git push -u origin master
   ```

### Step 3: Verify Upload

1. Go to your GitHub repository page
2. Confirm all files are uploaded
3. Check that catboost_info/ is ignored but model.pkl is included (for deployment)

## Notes

- The `model.pkl` file is excluded from version control as it can be regenerated by running `train.py`
- Training logs in `catboost_info/` are also excluded
- Make sure to install all required packages before running the application
- The application runs on port 5000 by default
