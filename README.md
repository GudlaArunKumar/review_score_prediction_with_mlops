# Ecommerce Review Score Prediction with mlops

**Problem statement**: For a given customer's historical data, we are tasked to predict the review score for the next order or purchase. We will be using the Brazilian E-Commerce Public Dataset by Olist. This dataset has information on 100,000 orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing charges from various dimensions: from order status, price, payment, freight performance to customer location, product attributes and finally, reviews written by customers. The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc

The main objective of this project is to implement a full stack data science workflow into it. To achieve that, I have used ZenML a Mlops pipeline infrastructure framweork.

The tech stack used in this project is as follows 

* Python - Programming language
* ZenML - MLOps framework
* MlFlow - Experiment tracking, logging & deployment flavor
* Streamlit - Deploying a web application for a quick PoC 

I have implemented the re-training framework where model can be re-trained from latest data updates by running an command in terminal and newly trained model will be deployed to ZenML server if it satisfies the minimum R2 Score and RMSE error. The following are the commands to perfom the operation.

To re-train the model - `python run_deployment.py --config deploy_and_predict`

For viewing a web application - `streamlit run streamlit_app.py`


**Note:**

ZenML deployment Server runs smoothly on MacOS or Linux OS but for Windows users, the turnaround is to use WSL (Windows Subystem for Linux) on Windows machine and run from any the installed bash terminal.