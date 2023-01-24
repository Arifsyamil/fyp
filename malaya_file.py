#Import all neccessary libraries
import streamlit as st
import re
import wikipediaapi
import malaya
import torch
import tensorflow
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc

st.cache(allow_output_mutation=True)
def malaya_model(model_name, kata):
	global df_malaya
	q_model = malaya.entity.transformer(model = model_name, quantized = True)
	malay_pred = q_model.predict(kata)
	df_malaya = pd.DataFrame(malay_pred, columns = ['kata', 'entiti'])
	return df_malaya