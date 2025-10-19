# Data Science Portfolio Projects

A comprehensive collection of data science and machine learning projects demonstrating various techniques and methodologies.

## Author
**Chaghoub Aymen**

## Projects Overview

This repository contains 5 distinct data science projects, each focusing on different domains and techniques:

### 1. 🗞️ Fake News Detection (`fake_news/`)
**Technique**: Text Classification, NLP  
**Models**: PassiveAggressiveClassifier  
**Features**: TF-IDF Vectorization, Text Preprocessing  
**Dataset**: News articles labeled as fake/true

### 2. 🎬 Netflix Content Analysis (`netflix/`)
**Technique**: Classification, Feature Engineering  
**Models**: Random Forest, Logistic Regression  
**Features**: Content popularity prediction based on metadata  
**Dataset**: Netflix Movies and TV Shows

### 3. 📊 Sales Dashboard (`sales/`)
**Technique**: Data Visualization, Interactive Dashboards  
**Technology**: Dash, Plotly  
**Features**: Real-time sales analytics and KPIs  
**Dataset**: Sales transaction data

### 4. 🏠 House Price Prediction (`housing/`)
**Technique**: Regression Analysis  
**Models**: Linear Regression, Ridge, Random Forest, Gradient Boosting  
**Features**: Comprehensive EDA, Model comparison, Cross-validation  
**Dataset**: USA Housing data

### 5. 🐦 Twitter Sentiment Analysis (`twitter/`)
**Technique**: NLP, Sentiment Classification  
**Models**: Naive Bayes, Logistic Regression, SVM  
**Features**: Text preprocessing, TF-IDF, WordClouds  
**Dataset**: Twitter airline sentiment data

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── fake_news/
│   │   └── fake_news_detection.py
│   ├── netflix/
│   │   └── netflix_analysis.py
│   ├── sales/
│   │   └── sales_dashboard.py
│   ├── housing/
│   │   └── house_price_prediction.py
│   └── twitter/
│       └── sentiment_analysis.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── models/
├── visualizations/
├── tests/
├── requirements.txt
└── README.md
```

## How to Run Projects

### 1. Fake News Detection
```bash
cd src/fake_news/
python fake_news_detection.py
```

### 2. Netflix Analysis
```bash
cd src/netflix/
python netflix_analysis.py
```

### 3. Sales Dashboard
```bash
cd src/sales/
python sales_dashboard.py
# Open http://localhost:8050 in your browser
```

### 4. House Price Prediction
```bash
cd src/housing/
python house_price_prediction.py
```

### 5. Twitter Sentiment Analysis
```bash
cd src/twitter/
python sentiment_analysis.py
```

## Key Features

- **Comprehensive EDA**: Detailed exploratory data analysis for each project
- **Multiple ML Models**: Comparison of various algorithms
- **Data Visualization**: Rich visualizations using matplotlib, seaborn, and plotly
- **Interactive Dashboards**: Real-time data visualization with Dash
- **NLP Processing**: Advanced text preprocessing and sentiment analysis
- **Model Persistence**: Save and load trained models
- **Cross-validation**: Robust model evaluation techniques

## Technologies Used

- **Python**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Web Dashboards**: Dash
- **NLP**: nltk, wordcloud
- **Machine Learning**: scikit-learn, various ML algorithms

## Skills Demonstrated

- Data preprocessing and cleaning
- Feature engineering
- Machine learning model development
- Data visualization
- Natural language processing
- Interactive dashboard creation
- Model evaluation and comparison
- Cross-validation techniques

## Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## License

This project is open source and available under the [MIT License](LICENSE).
