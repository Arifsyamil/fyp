#Import all neccessary libraries
import streamlit as st
import re
import wikipediaapi
import malaya
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

#Page header, title
st.set_page_config(page_title= "Malay Named Entity Recognition (NER) Model", page_icon= ":book:", layout= "wide")
st.title(":book: Malay Named Entity Recognition (NER) model")
st.markdown("Sila tekan butang di bawah untuk mulakan program")
btn_main = st.button("TEKAN MULA")
#1 BUTTON to go STEP 1 - STEP 8
#1 BUTTON for STEP 7: HYPERTUNING MODEL
#1 BUTTON for STEP 8: ERROR RATE
#1 BUTTON for FORM PREDICT WORDS

#LOAD PAGE AND GET TEXT
with st.container():
	st.write("---")
	left_col1, right_col1 = st.columns(2)
	with left_col1:
		st.header("LANGKAH 1: MUAT TURUN DATA DARIPADA WIKIPEDIA")
		st.markdown("1. Import wikipedia-api")
		st.markdown("2. Ekstrak teks daripada laman web Wikipedia")
		st.markdown("3. Simpan teks ke dalam fail malaytest.txt")
	with right_col1:
		@st.cache(allow_output_mutation=True)
		def find_text():
			global article, link, check
			mwiki = wikipediaapi.Wikipedia(language = 'ms', extract_format = wikipediaapi.ExtractFormat.WIKI)
			page = mwiki.page("Pemahsyuran Kemerdekaan Tanah Melayu")
			check = page.exists()
			link = page.fullurl
			article = page.text
			namefile = "malaytext.txt"
			with open(namefile, "w+", encoding='utf-8') as f:
				f.writelines(article)
			#st.write("Adakah URL ini wujud? ", page.exists())
			#st.write("URL Laman Web: ", link)
			#st.write("TEKS: ", "\n", article[:1000])
			return article, page
		if btn_main:
			result1 = find_text()
			st.write("Adakah URL ini wujud? ", check)
			st.write("URL Laman Web: ", link)
			st.write("TEKS: ", "\n", article[:1000])
			st.success("#1 FIND_TEXT: Selesai!")
		else:
			st.info("Belum ditekan")

#CLEAN DATA
with st.container():
	st.write("---")
	left_col2, right_col2 = st.columns(2)
	with left_col2:
		st.header("LANGKAH 2: PEMBERSIHAN DATA")
		st.markdown("1. Baca teks daripada malaytext.txt")
		st.markdown("2. Buang tanda baca daripada teks")
	with right_col2:
		def clean_data():
			global clean_file
			with open("malaytext.txt", "r", encoding='utf-8') as r:
				row = r.read()
			file = str(row)	
			file1 = file.strip("\n")
			file1 = re.sub("[=(),:;.]", "", file1)
			file1 = file1.strip()
			file1 = re.sub("[-']", " ", file1)
			file1 = file1.strip()
			file1 = file1.replace("\n", " ")
			#st.write("ORIGINAL FILE: ", "\n", article)
			clean_file = file1
			#st.write("CLEANED FILE: ", "\n", clean_file[:1000])
			return clean_file
		if btn_main:
			result2 = clean_data()
			st.write("CLEANED FILE: ", "\n", clean_file[:1000])
			st.success("#2 CLEAN_DATA: Selesai!")
		else:
			st.info("Belum ditekan")

#USE MALAYA MODULE
with st.container():
	st.write("---")
	left_col3, right_col3 = st.columns(2)
	with left_col3:
		st.header("LANGKAH 3 : IMPLEMENTASI MODUL MALAYA")
		st.markdown("1. Import modul malaya")
		st.markdown("2. Menggunakan transformer BERT")
		st.markdown("3. Menggunakan fungsi predict() bagi mengenal pasti jenis entity bagi setiap perkataan")
	with right_col3:
		def use_malaya():
			global malay_pred
			#model = malaya.entity.transformer(model= 'alxlnet')
			#q_model = malaya.entity.transformer(model = 'alxlnet', quantized = True)
			#model = quantized_model = malaya.entity.transformer(model = 'bert')
			q_model = malaya.entity.transformer(model1 = 'bert', quantized = True)
			malay_pred = q_model.predict(clean_file)
			#van = quantized_model.analyze(clean_file1)
			#st.table(malay_pred[:15])
			return malay_pred
		if btn_main:
			result3 = use_malaya()
			st.table(malay_pred[:15])
			st.success("#3 USE_MALAYA: Selesai!")
		else:
			st.info("Belum ditekan")

#ORGANISE DATAFRAME MODEL (NO ST.COLUMNS)
with st.container():
	st.write("---")
	st.header("LANGKAH 4 : ORGANISASI DATA MELALUI DATAFRAME")
	st.markdown("1. Kemaskini data melalui pd.DataFrame")
	st.markdown("2. Menambah lajur 'nombor', 'kata', 'entiti', 'SEBELUM', 'SELEPAS', 'TAGSEBELUM', 'TAGSELEPAS', 'LAIN-LAIN', 'LOKASI', 'MANUSIA', 'ORGANISASI', 'LKATA', 'LSEBELUM', 'LSELEPAS', 'LENTITI' ")
	st.markdown("3. Mengehadkan jenis entiti kepada LOKASI, MANUSIA, ORGANISASI dan LAIN-LAIN")
	st.markdown("4. Menukar nilai entiti kepada nilai binary melalui ONEHOTENCODER")
	st.markdown("4. Menukar nilai bukan bernombor(kata, entiti, SEBELUM, SELEPAS) kepada nilai bernombor melalui LABELENCODER")
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
		#st.table(df4.head())
		return df4
	if btn_main:
		result4 = data_model()
		st.table(df4.head())
		st.success("#4 DATA_MODEL : Selesai!" )
	else:
		st.info("Belum ditekan")

#TRAIN MODEL USING KNN, MULTIOUTPUTCLASSIFIER
with st.container():
	st.write("---")
	left_col4, right_col4 = st.columns(2)
	with left_col4:
		st.header("LANGKAH 5 : MELATIH MODEL")
		st.markdown("1. Mengenal pasti fitur dan kelas")
		st.markdown("2. Menembahagikan dataset latihan dan dataset ujian")
		st.markdown("3. Menggunakan model pembeljaran mesin KNN dan MultiOutputClassifier")
		st.markdown("4. Meramalkan nilai dataset x_test")
		st.markdown("5. Mengira skor model")
	with right_col4:
		def train_model():
			global x, y, y_test, y_pred, knn, classifier, model_score
			x = df4.iloc[:, [11,12,13]]
			#x
			y = df4.iloc[:,[8,9,10]]
			#y
			#from sklearn.model_selection import train_test_split
			x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42, stratify = y)
			#from sklearn.neighbors import KNeighborsClassifier
			knn = KNeighborsClassifier(n_neighbors= 4) #default 1st time k = 3, but entity type = 4
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
			#st.write('Predicted KNN:', "\n")
			#st.table(y_pred)
			#knn.score(y_test, y_pred)
			model_score = classifier.score(datay_test, y_pred)
			#st.write("Skor model KNN: ")
			#st.text(model_score)
			return x, y, y_test, y_pred, classifier, model_score
		#if btn_main:
			#result5 = train_model()
			#st.write('Predicted KNN:', "\n")
			#st.table(y_pred)
			#st.write("Skor model KNN: ")
			#st.text(model_score)
			#st.success("#5 TRAIN_MODEL: Selesai!")
		#else:
			#st.info("Belum ditekan")

#EVALUATE MODEL
with st.container():
	st.write("---")
	st.header("LANGKAH 6 : MENILAI PRESTASI MODEL")
	st.markdown("1. Mengubah suai struktur data y_test dan y_pred (data shape)")
	st.markdown("2. Menghasilkan confusion_matrix")
	st.markdown("3. Menghasilkan classification_report")
	st.markdown("4. Mengira accuracy_score")
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
	#if btn_main:
		#result6 = evaluate_model()
		#st.text("Confusion Matrix \n")
		#st.table(cm)
		#st.text("Classification Report \n")
		#st.text(cr)
		#st.write("Accuracy: ", "\n")
		#st.text(accuracy)
		#st.success("#6 EVALUATE_MODEL : Selesai!")
	#else:
		#st.info("Belum ditekan")


#HYPERTUNE MODEL
with st.container():
	global scores_list
	st.write("---")
	st.header("LANGKAH 7 : MENAMBAH BAIK MODEL")
	st.markdown("1. Membina model KNN dan MultiOutputClassifier dari nilai k = 1 hingga k = 50")
	st.markdown("2. Mengira nilai accuracy bagi setiap pertambahan nilai k")
	st.markdown("3. Menghasilkan graf")
	st.markdown("4. Mengira nilai maksimum accuracy pada nilai k tertentu")
	def hypertune_model():
		#from sklearn.model_selection import train_test_split
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42, stratify = y)
		k_range = range(1,51)
		scores = {}
		scores_list = []
		for k in k_range:
			knn2 = KNeighborsClassifier(n_neighbors = k)
			classifier2 = MultiOutputClassifier(knn2, n_jobs = -1)
			classifier2.fit(x_train, y_train)
			datay_test2 = y_test.values
			y_predict = classifier2.predict(x_test)
			scores[k] = classifier2.score(datay_test2, y_predict)
			scores_list.append(classifier2.score(datay_test2, y_predict))
		#for j in range(len(scores_list)):
			#st.write("k = ", j+1, ", Accuracy = ", scores_list[j])
		ht_df = pd.DataFrame(scores_list, columns=["score"])
		ht_df.index = np.arange(1,len(ht_df)+1)
		st.subheader("Skor accuracy bagi model KNN dari k = 1 hingga k = 50")
		#st.table(ht_df.head())
		st.dataframe(ht_df.transpose())
		st.text("Maksimum nilai k dan maksimum nilai accuracy")
		#st.write("Maksimum k = ", scores_list.index(max(scores_list))+1, ", Maksimum Accuracy = ", max(scores_list))
		st.text("Maksimum k = ").text(scores_list.index(max(scores_list))+1).text(", Maksimum Accuracy = ").text(max(scores_list))
		st.text("Purata nilai accuracy")
		st.write('Purata skor : {}'.format(np.mean(scores_list)))
		
		#import matplotlib.pyplot as plt
		fig1 = plt.figure(figsize = (3,3))
		plt.plot(k_range, scores_list)
		plt.xlabel("VALUE OF K FOR KNN MODEL")
		plt.ylabel("TESTING ACCURACY")
		st.pyplot(fig1)
		return scores_list
	#if btn_main:
		#result7 = hypertune_model()
		#st.success("#7 HYPERTUNE_MODEL : Selesai!")
	#else:
		#st.info("Belum ditekan")

#OVERFITTING & UNDERFITTING MODEL, COLLINEARITY
with st.container():
	global train_scores, test_scores
	st.write("---")
	st.header("LANGKAH 8 : OVERFITTING DAN UNDERFITTING MODEL")
	st.markdown("1. Membina model KNN dan MultiOutputClassifier dari nilai k = 1 hingga k = 50")
	st.markdown("2. Mengira nilai accuracy bagi training_dataset dan testing_dataset pada setiap kenaikan nilai k")
	st.markdown("3. Menghasilkan 2 graf bagi membandingkan trend graf training_dataset dan testing_dataset")
	def o_v_model():
		train_scores, test_scores = [], []
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42, stratify = y)
		values = [i for i in range(1, 51)]
		# evaluate a decision tree for each depth
		for i in values: 
			mod = KNeighborsClassifier(n_neighbors=i)
			cmodel = MultiOutputClassifier(mod, n_jobs = -1)
			# fit model
			cmodel.fit(x_train, y_train)
			# evaluate train dataset
			datay_train = y_train.values
			train_yhat = cmodel.predict(x_train)
			train_acc = cmodel.score(datay_train, train_yhat)
			train_scores.append(train_acc)
			# evaluate test dataset
			datay_test3 = y_test.values
			test_yhat = cmodel.predict(x_test)
			test_acc = cmodel.score(datay_test3, test_yhat)
			test_scores.append(test_acc)
			# summarize progress
			#st.write('>', i, ', train: ', train_acc, ', test: ', test_acc)
		st.subheader("Skor accuracy bagi training_dataset dan testing_dataset dari k = 1 hingga k = 50")
		test_train_data = {"Train Score" : train_scores, "Test Score" : test_scores}
		test_train_df = pd.DataFrame(test_train_data)
		test_train_df.index = np.arange(1, len(test_train_df)+1)
		st.table(test_train_df.transpose())
		# plot of train and test scores vs number of neighbors
		fig2 = plt.figure(figsize = (3,3))
		plt.plot(values, train_scores, '-o', label='Train')
		plt.plot(values, test_scores, '-o', label='Test')
		plt.legend()
		st.pyplot(fig2)
		return train_scores, test_scores
	#if btn_main:
		#result8 = o_v_model()
		#st.success("#8 O_V_MODEL : Selesai!")
	#else:
		#st.info("Belum ditekan")

#ERROR RATE
with st.container():
	global error
	st.write("---")
	st.header("LANGKAH 9 : KADAR RALAT (ERROR RATE)")
	st.markdown("1. Membina model KNN dan MultiOutputClassifier dari nilai k = 1 hingga k = 50")
	st.markdown("2. Mengira nilai error bagi setiap pertambahan nilai k pada model KNN")
	st.markdown("3. Menghasilkan graf bagi membuat pemerhatian trend graf nilai error")
	def error_rate():
		error = []
		counter = [j for j in range(1, 51)]
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42, stratify = y)
		for n in counter:
			mod1 = KNeighborsClassifier(n_neighbors=n)
			cmod1 = MultiOutputClassifier(mod1,  n_jobs = -1)
			cmod1.fit(x_train, y_train)
			error_pred = cmod1.predict(x_test)
			error_ytest = y_test.to_numpy() #convert DataFrame to Numpy array 
			error.append(np.mean(error_pred != error_ytest)) #comparison y_pred with y_test, for each entity, get its mean value
		error_df = pd.DataFrame(error, columns=["error_score"])
		error_df.index = np.arange(1,len(error_df)+1)
		#st.write(len(error_df))
		#st.dataframe(error_df.head())
		st.dataframe(error_df.transpose())

		fig3 = plt.figure(figsize = (3,3))
		plt.plot(counter, error, color='green', linestyle='dashed', marker='o', markerfacecolor='red', markersize=5)
		plt.title('Error Rate against K Value')
		plt.xlabel('K')
		plt.ylabel('Error Rate')
		st.pyplot(fig3)
		st.write("Minimum error:-", min(error), " at K = ", error.index(min(error))+1)
		return error
	#if btn_main:
		#result9 = error_rate()
		#st.success("#9 ERROR_RATE: Selesai!")
	#else:
		#st.info("Belum ditekan")
def load_model():
	article = find_text()
	clean_file = clean_data()
	malay_pred = use_malaya()
	df4 = data_model()
	x = df4.iloc[:, [11,12,13]]
	y = df4.iloc[:,[8,9,10]]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42, stratify = y)
	pred_knn = KNeighborsClassifier(n_neighbors= 4)
	#pred_knn.fit(x_train, y_train)
	#"classifier" VARIABLE from "TEST MODEL USING TESTING DATA"
	kelas = MultiOutputClassifier(pred_knn, n_jobs = -1)
	kelas.fit(x_train, y_train)
	st.write("LOAD MODEL:  BERJAYA")
	return article, clean_file, malay_pred, df4, x, y, x_train, x_test, y_train, y_test, pred_knn, kelas

#TO BE INSERTED below SUBMIT_BUTTON:
with st.container():
	st.write("---")
	st.header("LANGKAH 10 : RAMAL KATA LUAR DARIPADA DATASET")
	#st.markdown("1. Tulis sebuah ayat")
	#st.markdown("2. Buang semua tanda baca")
	#st.markdown("3. Memasukkan data ke dalam DataFrame")
	#st.markdown("4. Bina lajur LSEBELUM, LSELEPAS")
	#st.markdown("5. Mengehadkan jenis entity kepada LAIN-LAIN, LOKASI, MANUSIA, ORGANISASI")
	#st.markdown("6. Memasukkan data ke dalam DataFrame")
	@st.cache(suppress_st_warning=True)
	def ramal_kata():
		string = re.sub("[=(),:;.]", "", kata)
		string1 = string.split(" ")
		string2 = pd.DataFrame(string1, columns = ["LKATA"])
		string2['LSEBELUM'] = string2['LKATA'].shift(1)
		string2['LSELEPAS'] = string2['LKATA'].shift(-1)
		string2.fillna("null", inplace=True)
		#string1
		st.table(string1[:10])
		lbl = LabelEncoder()
		lbl_sen = lbl.fit_transform(string2['LKATA'])
		lbl_bef = lbl.fit_transform(string2['LSEBELUM'])
		lbl_aft = lbl.fit_transform(string2['LSELEPAS'])
		string2 = pd.DataFrame({'LKATA':lbl_sen, 'LSEBELUM': lbl_bef, 'LSELEPAS' : lbl_aft})
		st.dataframe(string2.head())
		
		#Train, test model
		article, clean_file, malay_pred, df4, x, y, x_train, x_test, y_train, y_test, pred_knn, kelas = load_model()
		hasil = kelas.predict(string2)
		st.write("PREDICT MODEL: BERJAYA")
		#st.write(hasil)
		st.write("Bilangan ayat: ", len(hasil))
		#for z in hasil:
			#st.write(z)
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
		global perkata
		perkata = [(key, value) for i, (key, value) in enumerate(zip(string1, fin))]
		output = pd.DataFrame({"kata" : string1,
								"entiti" : fin})
		st.dataframe(output.transpose())
		return perkata

#DOWNLOAD BUTTON
def convert_df(df:pd.DataFrame):
     return df.to_csv(index=False).encode('utf-8')

def convert_json(df:pd.DataFrame):
    result = df.to_json(orient="index")
    parsed = json.loads(result)
    json_string = json.dumps(parsed)
    return json_string


#CREATE TEXT FORM
with st.form(key= 'my_form'):
	kata = st.text_area(label="Sila taip teks atau ayat:", max_chars= 500)
	#kata = st.text_input(label="Sila taip teks atau ayat:", max_chars= 250)
	submit_button = st.form_submit_button(label= ":arrow_right: Buat Ramalan")

if submit_button:
    if re.sub(r'\s+','',kata)=='':
        st.error('Ruangan teks tidak boleh kosong.')

    elif re.match(r'\A\s*\w+\s*\Z', kata):
        st.error("Teks atau ayat mestilah sekurang-kurangnya dua patah perkataan.")
    
    else:
    	st.markdown("### Hasil Ramalan")
    	#result1 = find_text()
    	#result2 = clean_data()
    	#result3 = use_malaya()
    	#result4 = data_model()
    	#result5 = train_model()
    	#result6 = evaluate_model()
    	#result7 = hypertune_model()
    	#result8 = o_v_model()
    	#result9 = error_rate()
    	result10 = ramal_kata()
    	
    	st.success("Butang hantar berfungsi!")


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
