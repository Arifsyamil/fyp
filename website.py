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
from malaya_file import *
from knn_file import *

#Page header, title
st.set_page_config(page_title= "Malay Named Entity Recognition (NER) Model", page_icon= ":book:", layout= "wide")
st.title(":book: Pengecaman Entiti Nama Malay (NER) model")
st.markdown("CARA MENGGUNAKAN PROGRAM")
st.markdown("1. Sila taip sebuah ayat atau teks tidak melebihi 500 karakter di ruangan bawah")
st.markdown("2. Pilih model untuk melakukan proses pengecaman entiti nama (NER) berdasarkan teks")
st.markdown("3. Klik butang 'BUAT RAMALAN' bagi memulakan program")
st.markdown("4. Paparan bagi setiap kata serta jenis entiti akan dipaparkan pada bahagian 'HASIL RAMALAN'")

#CREATE TEXT FORM
with st.form(key= 'my_form'):
	global kata, btn_model, df1, df2
	kata = st.text_area(label="Sila taip teks atau ayat:", max_chars= 500)
	
	btn_model = st.radio("Pilih model untuk pengecaman entiti nama",
	("KNN", "BERT", "Tiny-BERT", "ALBERT", "Tiny-ALBERT", "XLNET", "ALXLNET", "FASTFORMER", "Tiny-FASTFORMER"))
	
	submit_button = st.form_submit_button(label= ":arrow_right: Buat Ramalan")
	
	if submit_button:
		if re.sub(r'\s+','',kata)=='':
			st.error('Ruangan teks tidak boleh kosong.')
			
		elif re.match(r'\A\s*\w+\s*\Z', kata):
			st.error("Teks atau ayat mestilah sekurang-kurangnya dua patah perkataan.")
		
		else:
			if btn_model == "KNN":
				st.write("Anda pilih model : KNN")
				#df1 = knn_model()
				#df2 = ramal_kata(kata)
			else:
				st.write("Anda pilih model transformer: ", btn_model)
				
			st.success("Butang hantar berfungsi!")

with st.container():
	st.write("---")
	st.header("Hasil Ramalan")
	st.subheader("Ayat asal")
	st.write("##")
	st.write(kata)
	patah = str(len(kata.split()))
	st.write("Bilangan perkataan : {}".format(patah))
	st.write("##")
	if btn_model == 'KNN':
		df = ramal_kata(kata)
		df_test = df.copy()
	else:
		df_test = malaya_model(btn_model, kata)
	
	entiti = sorted(df_test['entiti'].unique())
	pilih = st.multiselect('Jenis entiti', entiti, entiti)
	df_pilihan = df_test [ (df_test['entiti'].isin(pilih)) ]
	st.table(df_pilihan.style.set_properties(**{'background-color': 'white', 'color': 'black'}))

#About model
with st.expander("About this app", expanded=True):
	st.write(
		"""     
		-   **Pengecaman Nama Entiti Malay** adalah sebuah aplikasi pembelajaran mesin yang dibangunkan bagi mengecam entiti pada setiap token menggunakan modul MALAYA (Husein, 2018)
		-   Projek ini adalah tugasan Final Year Project bagi Ijazah Sarjana Muda di UKM
		-   Aplikasi ini ingin menentukan model terbaik yang boleh digunakan bagi dokumen teks subjek sejarah Bahasa Melayu
		-   Model ini mempunyai 3 fitur utama iaitu kata asal, kata sebelum dan kata selepas. Kelas yang disasarkan ialah LOKASI, MANUSIA dan ORGANISASI
		-   Maklumat lanjut boleh hubungi Muhd Arif Syamil bin Mohd Rahimi melalui e-mel a177313@siswa.ukm.edu.my atau 012-7049021 
		""")

process = psutil.Process(os.getpid())
mem_size = str((process.memory_info().rss)) # in bytes, divide by 1 billion to GB
mem_size_mb = str((process.memory_info().rss) / 1000000) 
mem_size_gb = str((process.memory_info().rss) / 1000000000)
st.write("Penggunaan memori: {} bytes or {} MB or {} GB".format(mem_size, mem_size_mb, mem_size_gb))
# Dokumen Pemasyhuran Kemerdekaan 1957 telah ditulis dalam dua bahasa iaitu bahasa Melayu yang ditulis dalam Jawi dan bahasa Inggeris - No PERSON, LOCATION, ORGANISATION
# Ketika mendarat di Lapangan Terbang Sungai Besi, tetamu kehormat telah disambut oleh Pesuruhjaya Tinggi British di Tanah Melayu, Sir Donald Charles MacGillivray dan Lady MacGillivray, Yang di-Pertuan Agong Tanah Melayu yang pertama, Tuanku Abdul Rahman diiringi Raja Permaisuri Agong dan Perdana Menteri Tanah Melayu yang pertama, Tunku Abdul Rahman.
# Kedudukan sebuah kereta yang terjunam ke dalam Sungai Maaw di Jeti Feri Tanjung Kunyit, Sibu, semalam, sudah dikenal pasti. Jurucakap Pusat Gerakan Operasi (PGO), Jabatan Bomba dan Penyelamat Malaysia (JBPM) Sarawak, berkata kedudukan Toyota Camry di dasar sungai itu dikesan anggota Pasukan Penyelamat Di Air (PPDA) yang melakukan selaman kelima, hari ini, pada jam 3.49 petang.