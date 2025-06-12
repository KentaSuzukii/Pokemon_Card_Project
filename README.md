# 🧠 PokéPrice Analyzer

A full-stack machine learning web application that analyzes Pokémon card images, evaluates card prices, and recommends better value options for collectors and investors.

## 🎯 Project Overview

Together with my classmates, I built a complete ML-powered app that allows users to:

- 🃏 **Upload Pokémon card images** to identify the card
- 📈 **Evaluate the true market value** using a regression model
- 💡 **Receive recommendations** for undervalued cards within a custom budget
- 🎨 **Enjoy an interactive UI** with a playful Pokémon-themed experience

---

## 👨‍💻 My Role

I led the development of **three key components**:

### 1. 📈 Card Evaluation (Regression Modeling)
- Trained a regression model to predict the *true market value* of Pokémon cards
- Used historical and metadata features (e.g., release year, rarity, condition)
- Compared predictions against current market prices to flag:
  - Overpriced cards (⚠️ Overestimated)
  - Bargains (💸 Underestimated)

### 2. 🧠 Card Recommendation System
- Built a smart system to suggest the **best-value cards** within a user’s specified budget
- Ranked cards based on **value-to-cost ratio**
- Helped users discover hidden gems in the Pokémon card market

### 3. 🎨 Front-End UI/UX (Streamlit)
- Designed and implemented the web interface using **Streamlit**
- Focused on **interactivity** and **visual appeal**
- Features included:
  - Pokémon-style background theme
  - //Master Ball// surprise
  - Iterative improvements based on user feedback from testing

---

## 🚀 Key Features

- 🧪 **ML Pipeline**: From data preprocessing to model deployment
- 🤝 **Team Collaboration**: Built under tight deadlines with strong teamwork
- 🌟 **User Delight**: Focused on both performance and playful experience

---

## 🛠️ Tech Stack

- **Python**: Core logic and ML models
- **Streamlit**: Front-end web interface
- **scikit-learn**: Regression modeling
- **pandas / NumPy**: Data processing
- **OpenCV / PIL**: Image handling (My teammate used it)
- **Git** + **GitHub**: Version control

---

## 📷 Screenshots
<img width="695" alt="Screenshot 2025-06-11 at 18 55 19" src="https://github.com/user-attachments/assets/c8f25936-1d5a-41b1-8be3-e84754b61529" />


---

## 📂 How to Run

```bash
# Clone the repo
git clone https://github.com/KentaSuzukii/Pokemon_Card_Project.git
cd pokeprice-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
