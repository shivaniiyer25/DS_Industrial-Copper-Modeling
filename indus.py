import streamlit as st
import pickle
import numpy as np
import sklearn
from streamlit_option_menu import option_menu

# Functions
def predict_status(ctry,itmtp,aplcn,wth,prdrf,qtlg,cstlg,tknslg,slgplg,itmdt,itmmn,itmyr,deldtdy,deldtmn,deldtyr):
    #change the datatypes "string" to "int"
    itdd = int(itmdt)
    itdm = int(itmmn)
    itdy = int(itmyr)

    dydd = int(deldtdy)
    dydm = int(deldtmn)
    dydy = int(deldtyr)
    #modelfile of the classification
    with open("C:/Users/shiva/Desktop/Guvi/Copper_Capstone/Classification_model.pkl", "rb") as f:
        model_class = pickle.load(f)

    user_data = np.array([[ctry, itmtp, aplcn, wth, prdrf, qtlg, cstlg, tknslg,
                           slgplg, itdd, itdm, itdy, dydd, dydm, dydy]])

    y_pred = model_class.predict(user_data)

    if y_pred == 1:
        return 1
    else:
        return 0

def predict_selling_price(ctry, sts, itmtp, aplcn, wth, prdrf, qtlg, cstlg,
                          tknslg, itmdt, itmmn, itmyr, deldtdy, deldtmn, deldtyr):
    #change the datatypes "string" to "int"
    itdd = int(itmdt)
    itdm = int(itmmn)
    itdy = int(itmyr)

    dydd = int(deldtdy)
    dydm = int(deldtmn)
    dydy = int(deldtyr)
    #modelfile of the classification
    with open("C:/Users/shiva/Desktop/Guvi/Copper_Capstone/Regression_Model.pkl", "rb") as f:
        model_regg = pickle.load(f)

    user_data = np.array([[ctry, sts, itmtp, aplcn, wth, prdrf, qtlg, cstlg, tknslg,
                           itdd, itdm, itdy, dydd, dydm, dydy]])

    y_pred = model_regg.predict(user_data)

    ac_y_pred = np.exp(y_pred[0])

    return ac_y_pred

st.set_page_config(layout="wide")

st.title(":violet[**INDUSTRIAL COPPER MODELING**]")
selected = option_menu(None,
                       options = ["Home","Analysis"],
                       default_index=0,
                       orientation="vertical",
                       styles={"container": {"width": "100%", "justify-content": "flex-start"},
                               "nav-link": {"font-size": "24px", "text-align": "center", "margin": "-2px"},
                               "nav-link-selected": {"background-color": "#6F36AD"}})
if selected == "Home":
    st.title(':violet[Overview]')
    st.markdown('''Problem Statement
    The copper industry faces challenges in managing sales and pricing data, which can often be skewed and noisy, leading to inaccurate manual predictions. Addressing these issues manually is both time-consuming and may not yield the best pricing decisions. A machine learning regression model can overcome these challenges by applying techniques such as data normalization, feature scaling, and outlier detection, along with using algorithms robust to skewed and noisy data.

    Another challenge in the copper industry is lead management. A lead classification model can evaluate and classify leads based on their likelihood of becoming customers. Using the STATUS variable, with WON indicating success and LOST indicating failure, you can filter out data points with other status values.

    The solution includes the following steps:

    Exploring skewness and outliers in the dataset.
    Transforming the data and performing necessary cleaning and preprocessing.
    Building a regression model to predict the continuous variable Selling_Price.
    Developing a classification model to predict lead status: WON or LOST.
    Creating a Streamlit page to input column values and receive the predicted Selling_Price or Status.
    About the Data
    id: Unique identifier for each transaction or item.
    item_date: Date of the transaction or item.
    quantity tons: Quantity of the item in tons.
    customer: Identifier for the customer.
    country: Country associated with each customer.
    status: Current status of the transaction or item.
    item type: Category of the items being sold or produced.
    application: Specific use or application of the items.
    thickness: Thickness of the items.
    width: Width of the items.
    material_ref: Reference identifier for the material used.
    product_ref: Reference identifier for the specific product.
    delivery date: Expected or actual delivery date for the item or transaction.
    selling_price: Price at which the items are sold.
    Approach
    Data Understanding
    Identify variable types (continuous, categorical) and their distributions.
    Convert invalid Material_Reference values starting with ‘00000’ to null.
    Treat reference columns as categorical variables.
    Exclude INDEX as it may not be useful.
    Data Preprocessing
    Handle missing values using mean, median, or mode.
    Treat outliers using IQR or Isolation Forest.
    Address skewness with appropriate data transformations, such as log or box-cox transformations.
    Encode categorical variables using suitable techniques like one-hot encoding or label encoding.
    Exploratory Data Analysis (EDA)
    Visualize outliers and skewness using boxplots, distplots, and violin plots.
    Feature Engineering
    Create new features if applicable and drop highly correlated columns using a heatmap.
    Model Building and Evaluation
    Split the dataset into training and testing sets.
    Train and evaluate different classification models using metrics such as accuracy, precision, recall, F1 score, and AUC.
    Optimize hyperparameters using cross-validation and grid search.
    Interpret model results and assess performance.
    Follow similar steps for regression modeling, noting that tree-based models may perform better due to noise and non-linearity.
    Model GUI
    Use Streamlit to create an interactive page.
    Provide task input (Regression or Classification).
    Create input fields for column values excluding Selling_Price for regression and Status for classification.
    Apply the same feature engineering and scaling as during model training.
    Display the prediction results.
    Tips
    Use the pickle module to save and load models (encoders, scalers, ML models).
    Fit and transform data separately and apply only the transform step for new data.
    Learning Outcomes
    Proficiency in Python and data analysis libraries (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit).
    Experience in data preprocessing techniques.
    Understanding of EDA techniques.
    Application of machine learning techniques for regression and classification.
    Building and optimizing ML models using evaluation metrics and techniques.
    Experience in feature engineering.
    Development of interactive web applications using Streamlit.
    Insights into manufacturing domain challenges and solutions through machine learning.
    This project equips you with practical skills in data analysis, machine learning modeling, and web application development, providing a solid foundation for solving real-world problems in the manufacturing domain.
    ''')

if selected == "Analysis":
    st.title(':violet[Industrial Copper Modeling]')

    tab1, tab2 = st.tabs(["Predict Status", "Estimate Selling Price"])

    with tab1:
            st.header("PREDICT STATUS (WON / LOST)")
            

            col1, col2 = st.columns(2)

            with col1:
                country = st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0", key='country1')
                item_type = st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0", key='item_type1')
                application = st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5", key='application1')
                width = st.number_input(label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0", key='width1')
                product_ref = st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579", key='product_ref1')
                quantity_tons_log = st.number_input(label="**Enter the Value for QUANTITY_TONS (Log Value)**/ Min:-0.322, Max:6.924", format="%0.15f", key='quantity_tons_log1')
                customer_log = st.number_input(label="**Enter the Value for CUSTOMER (Log Value)**/ Min:17.21910, Max:17.23015", format="%0.15f", key='customer_log1')
                thickness_log = st.number_input(label="**Enter the Value for THICKNESS (Log Value)**/ Min:-1.71479, Max:3.28154", format="%0.15f", key='thickness_log1')

            with col2:
                selling_price_log = st.number_input(label="**Enter the Value for SELLING PRICE (Log Value)**/ Min:5.97503, Max:7.39036", format="%0.15f", key='selling_price_log1')
                item_date_day = st.selectbox("**Select the Day for ITEM DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"), key='item_date_day1')
                item_date_month = st.selectbox("**Select the Month for ITEM DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"), key='item_date_month1')
                item_date_year = st.selectbox("**Select the Year for ITEM DATE**", ("2020", "2021"), key='item_date_year1')
                delivery_date_day = st.selectbox("**Select the Day for DELIVERY DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"), key='delivery_date_day1')
                delivery_date_month = st.selectbox("**Select the Month for DELIVERY DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"), key='delivery_date_month1')
                delivery_date_year = st.selectbox("**Select the Year for DELIVERY DATE**", ("2020", "2021", "2022"), key='delivery_date_year1')

            button = st.button(":violet[***PREDICT THE STATUS***]", use_container_width=True)

            if button:
                status = predict_status(country, item_type, application, width, product_ref, quantity_tons_log,
                                        customer_log, thickness_log, selling_price_log, item_date_day,
                                        item_date_month, item_date_year, delivery_date_day, delivery_date_month,
                                        delivery_date_year)

                if status == 1:
                    st.write("## :green[**The Status is WON**]")
                else:
                    st.write("## :red[**The Status is LOST**]")

    with tab2:
            st.header("**PREDICT SELLING PRICE**")
            st.write(" ")

            col1, col2 = st.columns(2)

            with col1:
                country = st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0", key='country2')
                status = st.number_input(label="**Enter the Value for STATUS**/ Min:0.0, Max:8.0", key='status')
                item_type = st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0", key='item_type2')
                application = st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5", key='application2')
                width = st.number_input(label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0", key='width2')
                product_ref = st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579", key='product_ref2')
                quantity_tons_log = st.number_input(label="**Enter the Value for QUANTITY_TONS (Log Value)**/ Min:-0.3223343801166147, Max:6.924734324081348", format="%0.15f", key='quantity_tons_log2')
                customer_log = st.number_input(label="**Enter the Value for CUSTOMER (Log Value)**/ Min:17.21910565821408, Max:17.230155364880137", format="%0.15f", key='customer_log2')

            with col2:
                thickness_log = st.number_input(label="**Enter the Value for THICKNESS (Log Value)**/ Min:-1.7147984280919266, Max:3.281543137578373", format="%0.15f", key='thickness_log2')
                item_date_day = st.selectbox("**Select the Day for ITEM DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"), key='item_date_day2')
                item_date_month = st.selectbox("**Select the Month for ITEM DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"), key='item_date_month2')
                item_date_year = st.selectbox("**Select the Year for ITEM DATE**", ("2020", "2021"), key='item_date_year2')
                delivery_date_day = st.selectbox("**Select the Day for DELIVERY DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"), key='delivery_date_day2')
                delivery_date_month = st.selectbox("**Select the Month for DELIVERY DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"), key='delivery_date_month2')
                delivery_date_year = st.selectbox("**Select the Year for DELIVERY DATE**", ("2020", "2021", "2022"), key='delivery_date_year2')

            button = st.button(":violet[***PREDICT THE SELLING PRICE***]", use_container_width=True)

            if button:
                price = predict_selling_price(country, status, item_type, application, width, product_ref, quantity_tons_log,
                                            customer_log, thickness_log, item_date_day,
                                            item_date_month, item_date_year, delivery_date_day, delivery_date_month,
                                            delivery_date_year)

                st.write("## :green[**The Selling Price is :**]", price)
