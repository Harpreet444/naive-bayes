import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import joblib


st.set_page_config(page_title='Naive Bayes',page_icon="🚥",layout='centered')
data_set = load_wine()

df = pd.concat([pd.DataFrame(data_set.data,columns=data_set.feature_names),pd.DataFrame(data_set.target,columns=['target'])],axis='columns')
x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'],axis='columns'),df['target'],random_state=10,test_size=0.2)



st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Naive Bayes</h1>", unsafe_allow_html=True)
st.write('''
1. Use wine dataset from sklearn.datasets to classify wines into 3 categories.
2. Load the dataset and split it into test and train.
3. After that train the model using Gaussian and Multinominal classifier and post which model performs better.
4. Use the trained model to perform some predictions on test data.
''')

# st.table(df.head())

gaussian = joblib.load('gaussian.job')
multinomial = joblib.load('multinomial.job')

col1,col2 = st.columns(2)
col1.markdown("<h2 style='text-align: center; color: #FF4B4B;'>Gaussian classifier</h2>", unsafe_allow_html=True)

col1.code("Accuracy: "+str(gaussian.score(x_test,y_test)))

fig, ax = plt.subplots()
cn_gaussian = confusion_matrix(y_test,gaussian.predict(x_test))
sns.heatmap(cn_gaussian,cmap='Reds',annot=True,xticklabels=data_set.target_names,yticklabels=data_set.target_names)
plt.title("Confusion matrix")
# ax.set_facecolor('#FDC8C8')
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predected Values")

col1.pyplot(fig)


col2.markdown("<h2 style='text-align: center; color: #FF4B4B;'>Multinomial classifier</h2>", unsafe_allow_html=True)

col2.code("Accuracy: "+str(multinomial.score(x_test,y_test)))

fig, ax = plt.subplots()
cn_multinomial = confusion_matrix(y_test,multinomial.predict(x_test))
sns.heatmap(cn_multinomial,cmap='Reds',annot=True,xticklabels=data_set.target_names,yticklabels=data_set.target_names)
plt.title("Confusion matrix")
# ax.set_facecolor('#FDC8C8')
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predected Values")

col2.pyplot(fig)

st.write("So, Gaussian classifier has better score than Multinomial classifier")

st.markdown("<h2 style='text-align: center; color: #FF4B4B;'>Model predictions</h2>", unsafe_allow_html=True)
co1,co2 = st.columns(2)

alcohol = co1.slider(label="alcohol", min_value=11.0, max_value=14.8, step=0.1)
malic_acid = co1.slider(label="malic_acid", min_value=0.7, max_value=5.8, step=0.1)
ash = co1.slider(label="ash", min_value=1.4, max_value=3.2, step=0.1)
alcalinity_of_ash = co1.slider(label="alcalinity_of_ash", min_value=10.6, max_value=30.0, step=0.1)
magnesium = co1.slider(label="magnesium", min_value=70.0, max_value=162.0, step=1.0)
total_phenols = co1.slider(label="total_phenols", min_value=1.0, max_value=3.9, step=0.1)
flavanoids = co1.slider(label="flavanoids", min_value=0.3, max_value=5.1, step=0.1)
nonflavanoid_phenols = co2.slider(label="nonflavanoid_phenols", min_value=0.1, max_value=0.7, step=0.1)
proanthocyanins = co2.slider(label="proanthocyanins", min_value=0.4, max_value=3.6, step=0.1)
color_intensity = co2.slider(label="color_intensity", min_value=1.3, max_value=13.0, step=0.1)
hue = co2.slider(label="hue", min_value=0.5, max_value=1.7, step=0.1)
od280_od315_of_diluted_wines = co2.slider(label="od280/od315_of_diluted_wines", min_value=1.3, max_value=4.0, step=0.1)
proline = co2.slider(label="proline", min_value=278.0, max_value=1680.0, step=10.0)

btn = co2.button(label="Predict")

if btn:
    output = data_set.target_names[gaussian.predict([[alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280_od315_of_diluted_wines, proline]])]
    co2.code(output)
