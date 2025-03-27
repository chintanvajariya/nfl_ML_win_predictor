# NFL Win Probability Dashboard

<table cellpadding="0" cellspacing="0" style="border: none;">
<tr>
<td width="71%" valign="top">

<p align="center">
  <img src="https://github.com/user-attachments/assets/3c0cbe12-a67b-4e85-a80a-4ac15132df34" alt="NFL Win Probability Dashboard" width="750"/>
</p>

[**Live App**](https://nflwinprobability.streamlit.app)

This interactive dashboard lets users explore **actual vs. predicted win probabilities** for NFL games. Built with Streamlit, it allows for dynamic adjustments to model parameters, feature selection, and intuitive game exploration — all in real time.

---

## What the Dashboard Does

- Visualizes predicted win probability throughout the game using an ML model  
- Compares model predictions with **actual win probability**  
- Displays **game outcome**, **key stats**, and **feature importance**  
- Lets users:  
  - Select a game by date, home/away teams  
  - Tune ML model settings like `n_estimators` and `max_depth`  
  - Customize features used in predictions (e.g. possession, time left, score)

<p align="center">
  <img src="https://github.com/user-attachments/assets/27153740-dcf3-4e89-a01c-d3bec0a4bc3d" alt="Predicted vs Actual Win Probabilities" width="750"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/6bb2781c-d6c1-4993-967c-223c69345fce" alt="Combined Win Probabilities" width="750"/>
</p>

---

## How It Works

- Uses the [`nfl_data_py`](https://pypi.org/project/nfl-data-py/) package to fetch play-by-play data  
- Trains a **Random Forest Regressor** to model win probability based on in-game features  
- Combines actual game data with model output for side-by-side comparison  
- Visualizes:
  - Predicted Win Probability
  - Actual Win Probability
  - Combined WP View
  - Model Error (MAE, MSE)
  - Feature Importance

<p align="center">
  <img src="https://github.com/user-attachments/assets/d2959dd2-6582-48e1-b535-f4632f0c144c" alt="Feature Importance" width="750"/>
</p>

---

## What I Learned From Building It

- How to move from a Jupyter Notebook prototype to a fully interactive web app  
- Gained real experience in **model interpretability**, **client-focused design**, and **data storytelling**  
- Learned the importance of visual feedback when debugging or improving ML models  
- Saw firsthand how **interactive dashboards** turn static insights into flexible tools

</td>

<td width="29%" valign="top" align="right">
  <img src="https://github.com/user-attachments/assets/8fe57f46-7e45-45d8-96f1-1946dbb89341" alt="Sidebar Banner" width="300"/>
  <img src="https://github.com/user-attachments/assets/ed2aad7a-2c56-4766-a87e-5eb536606535" alt="Sidebar Banner" width="300"/>
  <img src="https://github.com/user-attachments/assets/c9c766b7-c17b-4d74-b7d9-3599471a233e" alt="Sidebar Banner" width="300"/>
</td>
</tr>
</table>

## Broader Applications

The same techniques used here can be applied far beyond sports:

- **Finance**: Predicting stock movement or market behavior using historical data
- **Healthcare**: Patient risk scoring and treatment optimization
- **Manufacturing**: Predictive maintenance or defect detection  
- **Client-Facing Workflows**: Interactive dashboards for clients to explore models & scenarios  

## The core value? **Turning data + models into decisions.**
