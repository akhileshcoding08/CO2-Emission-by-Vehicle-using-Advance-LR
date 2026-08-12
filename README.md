# 🚗 CO2 Emission by Vehicle — EDA & Linear Regression

A complete data science project that explores the **Canada Vehicle CO2 Emissions dataset**, uncovers the key drivers of vehicle carbon emissions through exploratory data analysis (EDA), and builds multiple regression models to **predict CO2 emissions (g/km)** from a vehicle's technical specifications.

---

## 📌 Project Overview

Vehicle emissions are one of the largest contributors to greenhouse gases. This project analyzes **7,385 vehicle records** from Canada's official fuel consumption ratings dataset to answer two questions:

1. **What vehicle characteristics (engine size, cylinders, fuel type, fuel consumption, vehicle class, etc.) most strongly influence CO2 emissions?**
2. **Can we accurately predict a vehicle's CO2 emissions using machine learning regression models?**

The notebook walks through data cleaning, univariate/bivariate/multivariate visualization, correlation analysis, and finally trains and compares five regression models across three different train-test split ratios.

---

## 📂 Dataset

**Source file:** `CO2_Emissions_Canada.csv`

| Property | Details |
|---|---|
| Rows | 7,385 (7,385 → 6,282 after removing 1,103 duplicate rows) |
| Columns | 12 |
| Target Variable | `CO2 Emissions(g/km)` |

### Column Description

| Column | Description |
|---|---|
| `Make` | Vehicle brand |
| `Model` | Vehicle model |
| `Vehicle Class` | Vehicle class/category (SUV, Compact, Van, etc.) |
| `Engine Size(L)` | Engine displacement in litres |
| `Cylinders` | Number of engine cylinders |
| `Transmission` | Transmission type (A, AM, AS, AV, M) |
| `Fuel Type` | Fuel type (X = Regular gasoline, Z = Premium gasoline, E = Ethanol E85, D = Diesel, N = Natural Gas) |
| `Fuel Consumption City (L/100 km)` | City driving fuel consumption |
| `Fuel Consumption Hwy (L/100 km)` | Highway driving fuel consumption |
| `Fuel Consumption Comb (L/100 km)` | Combined (city + highway) fuel consumption |
| `Fuel Consumption Comb (mpg)` | Combined fuel consumption in miles per gallon |
| `CO2 Emissions(g/km)` | CO2 emissions — **target variable** |

---

## 🔍 Exploratory Data Analysis

The EDA is organized into three stages:

**Univariate Analysis**
- Distribution of vehicles by `Make`
- Distribution of fuel consumption (city/highway/combined) using violin plots

**Bivariate Analysis**
- CO2 emissions across vehicle classes (boxplot)
- Most common fuel types (bar chart)
- Share of total CO2 emissions by fuel type (pie chart)

**Multivariate Analysis**
- Fuel consumption vs. number of cylinders (stacked bar)
- Engine size vs. CO2 emissions, colored by fuel type (scatter plot)
- Engine size vs. CO2 emissions vs. cylinders vs. make (interactive bubble chart)
- Correlation heatmap of numeric features
- Brand → vehicle class → engine size/CO2 sunburst chart
- CO2 contribution by vehicle class (treemap)
- Pairplot of engine size, fuel consumption, and CO2 emissions

### Key Insights
- **Fuel consumption (combined)** and **engine size** are the strongest predictors of CO2 emissions (r ≈ 0.85–0.92).
- **Vans** (Passenger & Cargo) and **Standard SUVs/Pickups** are the highest per-vehicle CO2 emitters.
- **Engine Size** and **Cylinders** are highly correlated (r ≈ 0.93) — a redundant feature pair.
- **Regular** and **Premium gasoline** vehicles account for ~93% of records and ~92% of total emissions share.
- **Diesel** vehicles are more CO2-efficient per litre than gasoline vehicles of comparable engine size.

---

## 🤖 Modeling

After label-encoding categorical features, the dataset was split into features (`X`) and target (`y = CO2 Emissions(g/km)`), and evaluated across **3 train/test split ratios (60/40, 80/20, 90/10)** using **5 regression algorithms**:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- XGBoost Regressor

### Evaluation Metrics
Each model/split combination was scored using:
- **R² Score**
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

### Results Summary (90/10 split)

| Model | R² Score | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Linear Regression | 91.71% | 11.41 | 17.39 | 0.0448 |
| Ridge Regression | 91.71% | 11.41 | 17.39 | 0.0448 |
| Lasso Regression | 91.62% | 11.31 | 17.49 | 0.0442 |
| ElasticNet | 91.23% | 11.25 | 17.89 | 0.0437 |
| **XGBoost Regressor** | **99.75%** | **1.93** | **2.99** | **0.0078** |

🏆 **XGBoost Regressor** dramatically outperforms all linear models, achieving **~99.75% R²** with the lowest error across every metric — showing that the relationship between vehicle features and CO2 emissions, while largely linear, benefits significantly from a non-linear, ensemble-based model.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly Express |
| Machine Learning | Scikit-learn, XGBoost |
| Environment | Jupyter Notebook |

---

## 📁 Project Structure

```
CO2-Emission-by-Vehicle-EDA-and-LR/
│
├── CO2_Emission_by_Vehicle_EDA_and_LR.ipynb   # Main analysis & modeling notebook
├── CO2_Emissions_Canada.csv                   # Dataset
├── README.md                                  # Project documentation
└── requirements.txt                           # Python dependencies
```

---

## ⚙️ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/CO2-Emission-by-Vehicle-EDA-and-LR.git
   cd CO2-Emission-by-Vehicle-EDA-and-LR
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook CO2_Emission_by_Vehicle_EDA_and_LR.ipynb
   ```

5. Run all cells to reproduce the EDA visualizations and model results.

### `requirements.txt`
```
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
xgboost
lightgbm
jupyter
```

---

## 📈 Future Improvements
- Hyperparameter tuning (GridSearchCV / Optuna) for XGBoost to push performance further.
- Feature selection to remove redundant/highly-correlated features (Engine Size vs. Cylinders).
- Cross-validation instead of a single train-test split for more robust evaluation.
- Deploy the best model as a simple web app (Streamlit/Flask) for interactive CO2 prediction.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

## 📜 License
This project is open source and available under the [MIT License](LICENSE).

## 🙌 Acknowledgements
- Dataset: [Canada Government Open Data — Fuel Consumption Ratings](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)
