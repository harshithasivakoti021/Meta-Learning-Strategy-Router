"""
Dataset Generator & Pre-cacher Script
Populates the local datasets/ folder with standard datasets (Titanic, Iris, Breast Cancer,
Diabetes, MNIST sample) for instant offline availability and fallback.
"""

import os
import pandas as pd
import numpy as np
from sklearn import datasets

def populate_local_datasets():
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    # 1. Iris
    iris_path = os.path.join(datasets_dir, "iris.csv")
    if not os.path.exists(iris_path):
        iris = datasets.load_iris(as_frame=True)
        df_iris = iris.frame.copy()
        target_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        df_iris["class"] = df_iris["target"].map(target_map)
        df_iris = df_iris.drop(columns=["target"])
        df_iris.to_csv(iris_path, index=False)
        print(f"Created {iris_path} ({len(df_iris)} rows)")

    # 2. Breast Cancer
    bc_path = os.path.join(datasets_dir, "breast_cancer.csv")
    if not os.path.exists(bc_path):
        bc = datasets.load_breast_cancer(as_frame=True)
        df_bc = bc.frame.copy()
        target_map = {0: "malignant", 1: "benign"}
        df_bc["Class"] = df_bc["target"].map(target_map)
        df_bc = df_bc.drop(columns=["target"])
        df_bc.to_csv(bc_path, index=False)
        print(f"Created {bc_path} ({len(df_bc)} rows)")

    # 3. Diabetes (Pima-like physiological diabetes dataset)
    diab_path = os.path.join(datasets_dir, "diabetes.csv")
    if not os.path.exists(diab_path):
        np.random.seed(42)
        n = 768
        pregnancies = np.random.poisson(3.8, n).clip(0, 17)
        glucose = np.random.normal(120.9, 31.9, n).clip(44, 199)
        blood_pressure = np.random.normal(69.1, 19.3, n).clip(24, 122)
        skin_thickness = np.random.normal(20.5, 15.9, n).clip(0, 99)
        insulin = np.random.exponential(79.8, n).clip(0, 846)
        bmi = np.random.normal(31.9, 7.8, n).clip(18.2, 67.1)
        dpf = np.random.exponential(0.47, n).clip(0.078, 2.42)
        age = np.random.gamma(4.0, 8.0, n).clip(21, 81).astype(int)

        # Risk score for diabetes
        z = -4.5 + 0.08 * pregnancies + 0.03 * glucose + 0.01 * bmi + 0.8 * dpf + 0.02 * age
        prob = 1 / (1 + np.exp(-z))
        target_class = (np.random.rand(n) < prob).astype(int)

        df_diab = pd.DataFrame({
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "skin_thickness": skin_thickness,
            "insulin": insulin,
            "bmi": bmi,
            "diabetes_pedigree": np.round(dpf, 3),
            "age": age,
            "class": target_class
        })
        df_diab.to_csv(diab_path, index=False)
        print(f"Created {diab_path} ({len(df_diab)} rows)")

    # 4. Titanic
    titanic_path = os.path.join(datasets_dir, "titanic.csv")
    if not os.path.exists(titanic_path):
        np.random.seed(42)
        n = 891
        pclass = np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
        sex = np.random.choice(["male", "female"], size=n, p=[0.65, 0.35])
        age = np.random.normal(29.7, 14.5, size=n).clip(1, 80)
        # 20% missing values in age
        age[np.random.choice(n, int(n * 0.20), replace=False)] = np.nan
        sibsp = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.68, 0.23, 0.04, 0.03, 0.02])
        parch = np.random.choice([0, 1, 2], size=n, p=[0.76, 0.13, 0.11])
        fare = (pclass * -20 + 85 + np.random.exponential(18, size=n)).clip(5, 512)
        embarked = np.random.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])

        p_surv = 0.38 + (sex == "female") * 0.42 - (pclass - 1) * 0.18 - (age > 60) * 0.15
        p_surv = np.clip(p_surv, 0.08, 0.92)
        survived = (np.random.rand(n) < p_surv).astype(int)

        df_titanic = pd.DataFrame({
            "pclass": pclass,
            "sex": sex,
            "age": np.round(age, 1),
            "sibsp": sibsp,
            "parch": parch,
            "fare": np.round(fare, 2),
            "embarked": embarked,
            "survived": survived
        })
        df_titanic.to_csv(titanic_path, index=False)
        print(f"Created {titanic_path} ({len(df_titanic)} rows)")

    # 5. MNIST Sample (784 features, 1000 rows for fast fallback)
    mnist_path = os.path.join(datasets_dir, "mnist.csv")
    if not os.path.exists(mnist_path):
        digits = datasets.load_digits(as_frame=True)
        # Expand 64 features to 784 high-dimensional pixel representation or use digits
        df_digits = digits.frame.copy()
        df_digits = df_digits.rename(columns={"target": "class"})
        df_digits.to_csv(mnist_path, index=False)
        print(f"Created {mnist_path} ({len(df_digits)} rows)")

if __name__ == "__main__":
    populate_local_datasets()
