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

#Page header, title
st.set_page_config(page_title= "Malay Named Entity Recognition (NER) Model", page_icon= ":book:", layout= "wide")
st.title(":book: Malay Named Entity Recognition (NER) model")
st.markdown("Sila tekan butang di bawah untuk mulakan program")
btn_main = st.button("TEKAN MULA")
if btn_main:
	st.write("BERJAYA")
else:
	st.write("TIDAK TEKAN")

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
	#model = malaya.entity.transformer(model= 'alxlnet')
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
	#df.head()
	
	#shift(1) moves backward by 1
	df['SEBELUM'] = df['kata'].shift(1) #4
	#df
	#shift(-1) moves forward by 1
	df['SELEPAS'] = df['kata'].shift(-1) #5
	#df
	df['TAGSEBELUM'] = df['entiti'].shift(1) #6
	#df
	df['TAGSELEPAS'] = df['entiti'].shift(-1) #7
	#df
	#df['entiti'].unique()
	df.fillna("null", inplace=True)
	#st.table(df.head())
	#st.table(df.loc[df['entiti'] == 'location'])

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
	#df1
	#df1.loc[df['entiti'] == 'law']
	#st.table(df1['entiti'].value_counts())
	#st.write(df1.info())
	#st.table(df1.head())
	
	#ONE HOT ENCODER for LOKASI, MANUSIA dan ORGANISASI
	#from sklearn.preprocessing import OneHotEncoder
	ohe = OneHotEncoder()
	ohe_entity = ohe.fit_transform(df1[['entiti']]).toarray() #8, 9, 10, 11 Expected 4 entity type
	ohe_entity1 = pd.DataFrame(ohe_entity)
	df2 = df1.join(ohe_entity1)
	#st.table(df2.head())
	#st.write(df2['entiti'].unique())
	#df2.loc[df2['entiti'] == 'manusia']
	df2.columns = ['nombor', 'kata', 'entiti', 'SEBELUM', 'SELEPAS', 'TAGSEBELUM', 'TAGSELEPAS', 'LAIN-LAIN', 'LOKASI', 'MANUSIA', 'ORGANISASI']
	#st.write(df2.isnull().sum())

	#LABEL ENCODER for 'SEBELUM', 'SELEPAS', 'TAGSEBELUM', 'TAGSELEPAS', 
	#from sklearn.preprocessing import LabelEncoder
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
	#df4
	#df4.loc[df4['LENTITI'] == 3]
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
	#from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42, stratify = y)
	#from sklearn.neighbors import KNeighborsClassifier
	knn = KNeighborsClassifier(n_neighbors= 3) #default 1st time k = 3, but entity type = 4
	knn.fit(x_train, y_train)
	#from sklearn.multioutput import MultiOutputClassifier
	classifier = MultiOutputClassifier(knn, n_jobs = -1)
	classifier.fit(x_train, y_train)
	#st.write(x_train.shape)
	#st.write(x_test.shape)
	#st.write(y_train.shape)
	#st.write(y_test.shape)
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
	#y_test1
	y_pred1 = y_pred.flatten()
	#y_pred1
	#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
	cm = confusion_matrix(y_test1, y_pred1)
	#st.text("Confusion Matrix \n")
	#st.table(cm)
	cr = classification_report(y_test1, y_pred1)
	#st.text("Classification Report \n")
	#st.text(cr)
	accuracy = accuracy_score(y_test1, y_pred1)
	#st.write("Accuracy: ", "\n")
	#st.text(accuracy)
	return cm, cr, accuracy

st.cache(allow_output_mutation=True)
def knn_model():
	result1 = find_text()
	#st.write("Adakah URL ini wujud? ", page.exists())
	#st.write("URL Laman Web: ", link)
	#st.write("TEKS: ", "\n", article[:1000])
	st.success("#1 FIND_TEXT: Selesai!")
	result2 = clean_data()
	#st.write("CLEANED FILE: ", "\n", clean_file[:1000])
	st.success("#2 CLEAN_DATA: Selesai!")
	result3 = use_malaya()
	#st.table(malay_pred[:15])
	st.success("#3 USE_MALAYA: Selesai!")
	result4 = data_model()
	#st.table(df4.head())	
	st.success("#4 DATA_MODEL : Selesai!" )
	result5 = train_model()
	#st.write('Predicted KNN:', "\n")
	#st.table(y_pred)
	#st.write("Skor model KNN: ")
	#st.text(model_score)
	st.success("#5 TRAIN_MODEL: Selesai!")
	result6 = evaluate_model()
	#st.text("Confusion Matrix \n")
	#st.table(cm)
	#st.text("Classification Report \n")
	#st.text(cr)
	#st.write("Accuracy: ", "\n")
	#st.text(accuracy)
	st.success("#6 EVALUATE_MODEL : Selesai!")
	return result1, result2, result3, result4, result5, result6


st.cache(allow_output_mutation=True)
def malaya_model(model_name):
	global df_malaya
	#q_model = malaya.entity.transformer(model = model_name, quantized=True)
	q_model = malaya.entity.transformer(model = model_name, quantized=True)
	malay_pred = q_model.predict(kata)
	#df = st.dataframe(malay_pred)
	df_malaya = pd.DataFrame(malay_pred, columns = ['kata', 'entiti'])
	#st.dataframe(df_malaya)
	#st.success("DONE use_malaya")
	return df_malaya


#CREATE TEXT FORM
with st.form(key= 'my_form'):
	global kata, btn_model, df1, df2
	kata = st.text_area(label="Sila taip teks atau ayat:", max_chars= 500)
	
	btn_model = st.radio("Pilih model untuk pengecaman entiti nama",
		("KNN", "BERT", "Tiny-BERT", "ALBERT", "Tiny-ALBERT", "XLNET", "ALXLNET", "FASTFORMER", "Tiny-FASTFORMER"))
	
	if btn_model == 'KNN':
		st.write('Anda pilih model KNN.')
	elif btn_model == 'BERT':
		st.write('Anda pilih model BERT')
	elif btn_model == 'Tiny-BERT':
		st.write('Anda pilih model Tiny-BERT')
	elif btn_model == 'ALBERT':
		st.write('Anda pilih model ALBERT')
	elif btn_model == 'Tiny-ALBERT':
		st.write('Anda pilih model Tiny-ALBERT')
	elif btn_model == 'XLNET':
		st.write('Anda pilih model XLNET')
	elif btn_model == 'ALXLNET':
		st.write('Anda pilih model ALXLNET')
	elif btn_model == 'FASTFORMER':
		st.write('Anda pilih model FASTFORMER')
	elif btn_model == 'Tiny-FASTFORMER':
		st.write('Anda pilih model Tiny-FASTFORMER')												
	else:
		st.write("Sila pilih satu model untuk pengecaman entiti nama")

	submit_button = st.form_submit_button(label= ":arrow_right: Buat Ramalan")

	if submit_button:
		if re.sub(r'\s+','',kata)=='':
			st.error('Ruangan teks tidak boleh kosong.')
		elif re.match(r'\A\s*\w+\s*\Z', kata):
			st.error("Teks atau ayat mestilah sekurang-kurangnya dua patah perkataan.")
		else:
			st.markdown("### Hasil Ramalan")
			if btn_model == "KNN":
				st.write("USE KNN_MALAYA METHOD")
			else:
				st.write("you choose the correct model: ", btn_model)

		st.success("Butang hantar berfungsi!")


with st.container():
	st.write("---")
	st.markdown("### Hasil Ramalan")
	if btn_model == 'KNN':
		df1 = use_knn()
		st.write("KNN in progress", df1)
		st.success("DONE use_knn")
	else:
		df2 = malaya_model(btn_model)
		entiti = sorted(df2['entiti'].unique())
		pilih = st.multiselect('Jenis entiti', entiti, entiti)
		df_pilihan = df2 [ (df2['entiti'].isin(pilih)) ]
		st.write(kata)
		patah = str(len(kata.split()))
		st.write("Bilangan perkataan: {}".format(patah))
		df_line = df_pilihan.transpose()
		st.dataframe(df_line)
		new_df = df_line.style.set_properties(**{'background-color': 'white', 'color': 'black'})
		st.dataframe(new_df)

#df3 = use_malaya(btn_model)
#df_kata = list(df3['kata'])
#df_entiti = list(df3['entiti'])
#annotated_text(df_kata, df_entiti)
#choices = st.multiselect('Jenis entiti', choose_entity, choose_entity)
#df_selected_entity = df_malaya[(df_malaya['entiti'].isin(choices))]
#st.dataframe(df_selected_entity)

#About model
with st.expander("About this app", expanded=True):
    st.write(
        """     
-   **Pengecaman Nama Entiti Malay** adalah sebuah aplikasi pembelajaran mesin yang dibangunkan bagi mengecam entiti pada setiap token menggunankan modul MALAYA
-   Entiti yang dikaji ialah: LOKASI, MANUSIA, ORGANISASI 
-   Aplikasi ini menggunakan BERT yang mempunyai accuracy score yang paling tinggi dalam modul MALAYA
-   Model ini mempunyai 3 fitur utama iaitu kata, kata sebelum dan kata selepas. Kelas yang disasarkan adalah LOKASI, MANUSIA dan ORGANISASI
-   Maklumat lanjut boleh hubungi Muhd Arif Syamil melalui emel a177313@siswa.ukm.edu.my atau 012-7049021 
       """
    )

# Dokumen Pemasyhuran Kemerdekaan 1957 telah ditulis dalam dua bahasa iaitu bahasa Melayu yang ditulis dalam Jawi dan bahasa Inggeris - No PERSON, LOCATION, ORGANISATION
# Ketika mendarat di Lapangan Terbang Sungai Besi, tetamu kehormat telah disambut oleh Pesuruhjaya Tinggi British di Tanah Melayu, Sir Donald Charles MacGillivray dan Lady MacGillivray, Yang di-Pertuan Agong Tanah Melayu yang pertama, Tuanku Abdul Rahman diiringi Raja Permaisuri Agong dan Perdana Menteri Tanah Melayu yang pertama, Tunku Abdul Rahman.
# Kedudukan sebuah kereta yang terjunam ke dalam Sungai Maaw di Jeti Feri Tanjung Kunyit, Sibu, semalam, sudah dikenal pasti. Jurucakap Pusat Gerakan Operasi (PGO), Jabatan Bomba dan Penyelamat Malaysia (JBPM) Sarawak, berkata kedudukan Toyota Camry di dasar sungai itu dikesan anggota Pasukan Penyelamat Di Air (PPDA) yang melakukan selaman kelima, hari ini, pada jam 3.49 petang.
