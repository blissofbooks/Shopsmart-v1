# ShopSmart v5

A Python tool for retail customer segmentation and churn prediction.

## Overview

I built this project over a one-week sprint. The idea was to treat it like a data consulting job where the goal is to turn raw sales data into a strategy.

Instead of just making charts, the script uses machine learning to group customers and predict who is likely to stop buying. I also added a Gradio interface so you can use it without writing code.

## Key Features

1. Customer Segmentation
It uses RFM analysis to group customers based on how they actually buy. This helps separate the VIPs from the one-time shoppers.

2. Churn Prediction
The script looks at purchase history to flag customers who are at risk of leaving. It gives you a list of people to target before they churn.

3. Gradio Interface
I wrapped the logic in a web interface. You can upload a dataset and see the results instantly in the browser.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Gradio
- Matplotlib

## How to Run

1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ShopSmart-v5.git

2. Install the requirements
pip install -r requirements.txt

3. Run the app
python app.py

4. Open the local link shown in the terminal (usually http://127.0.0.1:7860).

## Team

I built this with support from my team. I handled the code, but they helped a lot with the strategy and logic.

- Aarav Jain
- Malhar Patnakar
- Anadi vijay Pathak
- Anusree krumana
- C Aarthi
- Ivana Ravuri