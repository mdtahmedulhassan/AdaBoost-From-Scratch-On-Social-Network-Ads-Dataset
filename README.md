# AdaBoost From Scratch 🚀

A step-by-step implementation of the AdaBoost algorithm built from scratch using Python — no sklearn AdaBoostClassifier used.

## 📌 About
This project implements AdaBoost manually using Decision Stumps as weak learners on the Social Network Ads dataset. The goal is to understand the internal math and logic behind the algorithm.

## 📂 Dataset
**Social Network Ads**
| Column | Description |
|---|---|
| Age | Age of the user |
| EstimatedSalary | Annual salary of the user |
| Purchased | 0 = Not Purchased, 1 = Purchased |

## 🧠 What's Implemented
- Uniform weight initialization
- Decision Stump training (weak learner)
- Weighted error calculation
- Alpha (model weight) calculation
- Sample weight update and normalization
- Weighted resampling using cumulative sum
- Final prediction using weighted majority vote

## 📐 Key Math
| Formula | Description |
|---|---|
| `ε = Σ wᵢ · 𝟙[yᵢ ≠ hₜ(xᵢ)]` | Weighted error |
| `α = 0.5 · ln((1−ε)/ε)` | Model weight (alpha) |
| `H(x) = sign(Σ αₜ · hₜ(x))` | Final prediction |

## 🛠️ Libraries Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- MLxtend
- Scikit-learn (only for Decision Tree as weak learner)

## 📁 File Structure
```
AdaBoost-From-Scratch/
│
├── Adaboost_From_Scratch.ipynb
├── Social_Network_Ads.csv
└── README.md
```

## ▶️ How to Run
1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/AdaBoost-From-Scratch.git
```
2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend
```
3. Open the notebook
```bash
jupyter notebook Adaboost_From_Scratch.ipynb
```

## 👤 Author
Your Name — [GitHub](https://github.com/YOUR_USERNAME) | [Kaggle](https://www.kaggle.com/YOUR_USERNAME)
