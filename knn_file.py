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
import os
import psutil

#LOAD PAGE AND GET TEXT
st.cache(suppress_st_warning=True)
def find_text():
	global article, link, page
	mwiki = wikipediaapi.Wikipedia(language = 'ms', extract_format = wikipediaapi.ExtractFormat.WIKI)
	page = mwiki.page("Pemahsyuran Kemerdekaan Tanah Melayu")
	link = page.fullurl
	article = page.text
	namefile = "malaytext.txt"
	return article, page, link

#CLEAN DATA
st.cache(suppress_st_warning=True)
def clean_data():
	global clean_file
	file = article
	file1 = file.strip("\n")
	file1 = re.sub("[=(),:;.]", "", file1)
	file1 = file1.strip()
	file1 = re.sub("[-']", " ", file1)
	file1 = file1.strip()
	file1 = file1.replace("\n", " ")
	clean_file = file1
	return clean_file

#USE MALAYA MODULE
st.cache(allow_output_mutation=True)
def use_malaya():
	global malay_pred
	q_model = malaya.entity.transformer(model1 = 'bert', quantized = True)
	malay_pred = q_model.predict(clean_file)
	return malay_pred

#ORGANISE DATAFRAME MODEL (NO ST.COLUMNS)
st.cache(allow_output_mutation=True)
def data_model():
	global df4 #Start as LABELENCODER
	df = pd.DataFrame(malay_pred)
	df.columns = ['kata', 'entiti'] #1, #2
	df['kata'].astype('str') #KIV
	df['entiti'].astype('str')
	df['nombor'] = df.reset_index().index #3
	df = df.reindex(['nombor', 'kata', 'entiti'], axis = 1)
	
	#shift(1) moves backward by 1
	df['SEBELUM'] = df['kata'].shift(1) #4
	#shift(-1) moves forward by 1
	df['SELEPAS'] = df['kata'].shift(-1) #5
	df['TAGSEBELUM'] = df['entiti'].shift(1) #6
	df['TAGSELEPAS'] = df['entiti'].shift(-1) #7
	df.fillna("null", inplace=True)

	#Observe entity LAIN-LAIN if it is a nuisance or otherwise
	df1 = df.copy()
	df1.replace("time", "OTHER", inplace=True)
	df1.replace("event", "OTHER", inplace=True)
	df1.replace("law", "OTHER", inplace=True)
	df1.replace("quantity", "OTHER", inplace=True)
	df1.replace("location", "lokasi", inplace=True)
	df1.replace("organization", "organisasi", inplace=True)
	df1.replace("person", "manusia", inplace=True)
	df1.replace("OTHER", "LAIN-LAIN", inplace=True)

	#ONE HOT ENCODER for LOKASI, MANUSIA dan ORGANISASI
	ohe = OneHotEncoder()
	ohe_entity = ohe.fit_transform(df1[['entiti']]).toarray() #8, 9, 10, 11 Expected 4 entity type
	ohe_entity1 = pd.DataFrame(ohe_entity)
	df2 = df1.join(ohe_entity1)
	df2.columns = ['nombor', 'kata', 'entiti', 'SEBELUM', 'SELEPAS', 'TAGSEBELUM', 'TAGSELEPAS', 'LAIN-LAIN', 'LOKASI', 'MANUSIA', 'ORGANISASI']

	#LABEL ENCODER for 'SEBELUM', 'SELEPAS', 'TAGSEBELUM', 'TAGSELEPAS', 
	le = LabelEncoder()
	le_word = le.fit_transform(df1['kata'])
	le_word1 = pd.DataFrame(le_word)
	df3 = df2.join(le_word1) #COLUMNS OVERLAPPED
	df3.columns = ['nombor', 'kata', 'entiti', 'SEBELUM', 'SELEPAS', 'TAGSEBELUM', 'TAGSELEPAS','LAIN-LAIN', 'LOKASI', 'MANUSIA', 'ORGANISASI', 'LKATA']
	le_before = le.fit_transform(df1['SEBELUM'])
	le_before1 = pd.DataFrame(le_before)
	df3 = df3.join(le_before1)
	df3.columns = ['nombor', 'kata', 'entiti', 'SEBELUM', 'SELEPAS', 'TAGSEBELUM', 'TAGSELEPAS', 'LAIN-LAIN', 'LOKASI', 'MANUSIA', 'ORGANISASI', 'LKATA', 'LSEBELUM']
	le_after = le.fit_transform(df1['SELEPAS'])
	le_after1 = pd.DataFrame(le_after)
	df4 = df3.join(le_after1)
	df4.columns = ['nombor', 'kata', 'entiti', 'SEBELUM', 'SELEPAS', 'TAGSEBELUM', 'TAGSELEPAS', 'LAIN-LAIN', 'LOKASI', 'MANUSIA', 'ORGANISASI', 'LKATA', 'LSEBELUM', 'LSELEPAS']
	le_entity = le.fit_transform(df1['entiti'])
	le_entity1 = pd.DataFrame(le_entity)
	df4 = df4.join(le_entity1)
	df4.columns = ['nombor', 'kata', 'entiti', 'SEBELUM', 'SELEPAS', 'TAGSEBELUM', 'TAGSELEPAS', 'LAIN-LAIN', 'LOKASI', 'MANUSIA', 'ORGANISASI', 'LKATA', 'LSEBELUM', 'LSELEPAS', 'LENTITI']
	
	df4['LKATA'] = df4['LKATA'].astype(str)
	df4['LSEBELUM'] = df4['LSEBELUM'].astype(str)
	df4['LSELEPAS'] = df4['LSELEPAS'].astype(str)
	df4['LAIN-LAIN'] = df4['LAIN-LAIN'].astype(int)
	df4['LOKASI'] = df4['LOKASI'].astype(int)
	df4['ORGANISASI'] = df4['ORGANISASI'].astype(int)
	df4['MANUSIA'] = df4['MANUSIA'].astype(int)
	return df4

#TRAIN MODEL USING KNN, MULTIOUTPUTCLASSIFIER
st.cache(allow_output_mutation=True)
def train_model():
	global x, y, y_test, y_pred, knn, classifier, model_score
	x = df4.iloc[:, [11,12,13]]
	y = df4.iloc[:,[8,9,10]]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42, stratify = y)
	knn = KNeighborsClassifier(n_neighbors= 3) #default 1st time k = 3, but entity type = 4
	knn.fit(x_train, y_train)
	classifier = MultiOutputClassifier(knn, n_jobs = -1)
	classifier.fit(x_train, y_train)
	#datax_test = x_test.values
	datay_test = y_test.values
	y_pred = classifier.predict(x_test)
	model_score = classifier.score(datay_test, y_pred)
	return x, y, y_test, y_pred, classifier, model_score

#EVALUATE MODEL
st.cache(allow_output_mutation=True)
def evaluate_model():
	global cm, cr, accuracy
	y_test1 = y_test.to_numpy().flatten()
	y_pred1 = y_pred.flatten()
	cm = confusion_matrix(y_test1, y_pred1)
	cr = classification_report(y_test1, y_pred1)
	accuracy = accuracy_score(y_test1, y_pred1)
	return cm, cr, accuracy

#LOAD MODEL
st.cache(allow_output_mutation=True)
def knn_model():
	result1 = find_text()
	result2 = clean_data()
	result3 = use_malaya()
	result4 = data_model()
	result5 = train_model()
	result6 = evaluate_model()
	return result1, result2, result3, result4, result5, result6

#PREDICT WORD OUTSIDE DATA
st.cache(allow_output_mutation=True)
def ramal_kata(kata):
	string = re.sub("[=(),:;.]", "", kata)
	string1 = string.split(" ")
	string2 = pd.DataFrame(string1, columns = ["LKATA"])
	string2['LSEBELUM'] = string2['LKATA'].shift(1)
	string2['LSELEPAS'] = string2['LKATA'].shift(-1)
	string2.fillna("null", inplace=True)
	#string1
	#st.table(string1[:10])
	lbl = LabelEncoder()
	lbl_sen = lbl.fit_transform(string2['LKATA'])
	lbl_bef = lbl.fit_transform(string2['LSEBELUM'])
	lbl_aft = lbl.fit_transform(string2['LSELEPAS'])
	string2 = pd.DataFrame({'LKATA':lbl_sen, 'LSEBELUM': lbl_bef, 'LSELEPAS' : lbl_aft})
	#st.dataframe(string2.head())
	
	#Train, test model
	pred_outdata = knn_model()
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42, stratify = y)
	pred_knn = KNeighborsClassifier(n_neighbors= 3)
	#"classifier" VARIABLE from "TEST MODEL USING TESTING DATA"
	kelas = MultiOutputClassifier(pred_knn, n_jobs = -1)
	kelas.fit(x_train, y_train)
	hasil = kelas.predict(string2)
	#st.write(hasil)
	
	fin = []
	for z in hasil:
		if (z == [1, 0, 0]).all():
			fin.append("LOKASI")
		elif (z == [0, 1, 0]).all():
			fin.append("MANUSIA")
		elif (z == [0, 0, 1]).all():
			fin.append("ORGANISASI")
		else:
			fin.append("LAIN-LAIN")
			
	#st.write(fin)
	global perkata, output
	perkata = [(key, value) for i, (key, value) in enumerate(zip(string1, fin))]
	output = pd.DataFrame({"kata" : string1, "entiti" : fin})
	#st.dataframe(output.transpose())
	return output

def get_data():
	ts = output
	return ts
