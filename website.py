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

st.cache(allow_output_mutation=True)
def use_malaya(model_name):
	global df_malaya
	#q_model = malaya.entity.transformer(model = model_name, quantized=True)
	q_model = malaya.entity.transformer(model = model_name)
	malay_pred = q_model.predict(kata)
	#df = st.dataframe(malay_pred)
	df_malaya = pd.DataFrame(malay_pred, columns = ['kata', 'entiti'])
	#st.dataframe(df_malaya)
	#st.success("DONE use_malaya")
	return df_malaya


st.cache(suppress_st_warning=True)
def use_knn():
	mode = 12
	return mode


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
	st.markdown("Hasil Rumusan")
	if btn_model == 'KNN':
		df1 = use_knn()
		st.write("KNN in progress", df1)
		st.success("DONE use_knn")
	else:
		df2 = use_malaya(btn_model)
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
