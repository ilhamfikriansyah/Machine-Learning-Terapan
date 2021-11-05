# Laporan Proyek Machine Learning – Ilham Fikriansyah

## Domain Proyek
Proyek yang dirancang kali ini merupakan permasalahan telekomunikasi pada Airline Travel Information System (ATIS) dimana banyaknya pertanyaan seputar airline travel system memiliki pertanyaan yang sama dan berulang ulang. Sehingga seluruh pertanyaan tersebut memiliki jawaban yang serupa, dari jawaban yang serupa tersebut dapat diklasifikasikan menjadi beberapa ketegori yang sama.

Masalah tersebut dapat diselesaikan dengan cara penggunaan Teknik Natural Language Processing dengan penggunaan chatbot, chatbot disini merupakan mesin yang dapat mengerti pesan berupa teks untuk dapat memberikan jawaban dari kategori yang tepat sesuai dengan pertanyaan yang dimaksud.


## Business Understanding
Solusi yang akan digunakan adalah Teknik Natural Language Processing menggunakan algoritma Support Vector Machine dengan alur sebagai berikut.

### Problem Statements
Bagaimana cara mengklasifikasikan teks berdasarkan pertanyaan yang masuk dan memberikan jawaban berupa kategori yang tepat berdasarkan ketentuan dari Airline Travel Information System?

### Goals
Membuat chatbot atau machine yang dapat memprediksi kategori dari pertanyaan teks yang masuk berdasarkan kategori dari Airline Travel Information System yang tersedia secara otomatis.


### Solution statements
Support Vector Machine (SVM) pertama kali diperkenalkan oleh Vapnik pada tahun 1992 sebagai rangkaian harmonis konsep-konsep unggulan dalam bidang pattern recognition. SVM adalah algoritma machine learning yang bekerja atas prinsip Structural Risk Minimization (SRM) dengan  tujuan  menemukan hyperplane terbaik  yang  memisahkan  dua buah class pada input space. SVM merupakan model ML multifungsi yang dapat digunakan untuk menyelesaikan permasalahan klasifikasi, regresi, pendeteksian outlier, dan termasuk ke dalam kategori supervised learning. 

Beberapa keunggulan Support Vector Machine sebagai berikut.
1.	SVM efektif pada data berdimensi tinggi (data dengan jumlah fitur atau atribut yang sangat banyak). 
2.	SVM efektif pada kasus di mana jumlah fitur pada data lebih besar dari jumlah sampel. 
3.	SVM menggunakan subset poin pelatihan dalam fungsi keputusan (disebut support vector) sehingga membuat penggunaan memori menjadi lebih efisien.

## Data Understanding
Dataset Airline Travel Information System (ATIS) adalah data yang digunakan dalam melatih pengklasifikasian yang bertujuan untuk mengetahui maksud pelanggan saat melakukan pertanyaan ataupun komentar. Datasets tersebut dapat didapatkan di [Airline Travel Information System | Kaggle](https://www.kaggle.com/hassanamin/atis-airlinetravelinformationsystem)

Variabel-variabel pada Airline Travel Information System dataset adalah sebagai berikut:

1.	Label : Berisi target atau class dari setiap kategori pertanyaan atau komentear dari Sentence, class tersebut terdiri dari 8 kategori yaitu atis_flight, atis_flight_time, atis_airfare, atis_aircraft, atis_ground_service, atis_airline, atis_abbreviation, dan atis_quantity

2.	Sentence : Berisi kalimat pertanyaan atau komentar yang telah diberi label yang sesuai mengenai Airline Travel Information System


## Data Preparation
Teknik yang digunakan pada tahapan Data Preparation adalah vektorisasi, vektorisasi adalah proses mengekstrak fitur dengan TF-IDF, TF-IDF adalah proses scaling berdasarkan seberapa sering kata tersebut muncul pada teks dan seluruh dokumen, sehingga kita dapat menghilangkan nilai pada data yang terlalu sering muncul dan yang jarang sekali muncul. Pada tahap ini parameter yang digunakan sebagai berikut. 

1.	min_df = 5 artinya mengabaikan term yang uncul dalam kurang dari 5 teks
2.	max_df = 0.8 artinya kita mengabaikan term yang muncul lebih dari 80% dalam teks 
3.	sublinear_tf = True artinya mengubah vektor frekuensi menjadi bentuk logaritmik (1+log(tf)) sehingga dapat menormalisasi bias terhadap teks yang panjang dan teks yang pendek.
4.	use_idf = True artinya memungkinkan kita untuk menggunakan Inverse Document Frequency (IDF). Hal ini berarti term yang terlalu sering muncul dalam teks akan diberi skor lebih sedikit dibanding term yang jarang muncul (hanya muncul pada teks yang spesifik saja)


Pada proses vektorisasi ini, digunakan dua metode sebagai berikut. 

1.	fit_transform Metode ini mempelajari kosa kata dan Inverse Document Frequency (IDF) dengan memberikan nilai return berupa document-term matrix. 
2.	transform Metode ini mentransformasi dokumen ke dalam document-term matrix.

Tujuan dari vektorisasi adalah mengekstrak teks mentah (raw text) dari data masukan, menghapus semua informasi yang tidak diperlukan, dan mengonversi teks ke dalam format yang dibutuhkan. Sehingga, teks siap dimasukkan ke dalam sistem Natural Language Processing.

## Modeling
Model machine learning yang digunakan adalah model machine learning dengan teknik Support Vector Machine (SVM) dengan kernel linear. algoritma machine learning ini bekerja atas prinsip Structural Risk Minimization (SRM) dengan  tujuan  menemukan hyperplane terbaik Dengan proses sebagai berikut.

1.	SVM mencari support vector pada setiap kelas. Support vector adalah sampel dari masing-masing kelas yang memiliki jarak paling dekat dengan sampel kelas lainnya.

2.	Setelah support vector ditemukan, SVM menghitung margin. Margin bisa kita anggap sebagai jalan yang memisahkan sejumlah kelas dalam kasus linear margin yang terbentuk merupakan margin yang berbentuk garis lurus atau linear. Margin ini dibuat berdasarkan support vector di mana support vector bekerja sebagai batas tepi jalan, atau sering kita kenal sebagai bahu jalan. SVM mencari margin terbesar atau jalan terlebar yang mampu memisahkan kedua kelas.

3.	Setelah menemukan jalan terlebar maka decision boundary dapat ditemukan, decision boundary adalah garis yang membagi jalan atau margin menjadi bagian yang sama besar. 

4.	Dari decision boundary maka Hyperplane dapat ditemukan. Hyperplane adalah bidang yang memisahkan kelas berbeda.

Dari Model tersebut didapatkan performa sebagai berikut.

                     precision    recall  f1-score   support

           accuracy                           0.98       800
          macro avg       0.87      0.98      0.91       800
       weighted avg       0.98      0.98      0.98       800


## Evaluation
Confusien matrix dari model yang telah dibuat adalah sebagai berikut.
Confusion Matrix
        [[ 33,   0,   0,   0,   0,   0,   0,   0],
        [  0,   8,   0,   0,   0,   0,   0,   1],
        [  0,   0,  47,   0,   1,   0,   0,   0],
        [  0,   0,   0,  37,   1,   0,   0,   0],
        [  0,   3,   3,   3, 619,   0,   0,   4],
        [  0,   0,   0,   0,   0,   1,   0,   0],
        [  0,   0,   0,   0,   0,   0,  36,   0],
        [  0,   0,   0,   0,   0,   0,   0,   3]]

Confusion matrix adalah pengukuran performa untuk masalah klasifikasi machine learning dimana keluaran dapat berupa dua kelas atau lebih. Plot dari true labels (columns) dan predicted label (rows) dengan urutan labels untuk columns dan rows sebagai berikut. 

labels = ['atis_abbreviation', 'atis_aircraft', 'atis_airfare', 'atis_airline','atis_flight', 'atis_flight_time', 'atis_ground_service', 'atis_quantity']

Dalam hal ini terdapat 8 true label dan 8 predicted label dari confusien matrix. Hal ini dikarenakan terdapat 8 kategori untuk di prediksi yang mewakili sebagai berikut.


1. True Positive (TP) : memprediksi positif dan itu benar
2. True Negative (TN) : memprediksi negatif dan itu benar
3. False Positive (FP) : memprediksi positif dan itu salah
4. False Negative (FN) : memprediksi negatif dan itu salah

Classification Metrics yang digunakan pada Airline Travel Information System adalah accuracy F1-score. 

F1-score adalah metrik yang menggabungkan precision dan recall. Dengan rumus 
F1 Score = 2 * (Recall*Precission) / (Recall + Precission)

Precision adalah metrik yang membandingkan rasio prediksi benar atau positif dengan keseluruhan hasil yang diprediksi positif dengan rumus
Precission = (TP) / (TP+FP), sedangkan

Recall adalah metrik yang membandingkan rasio prediksi benar atau positif dengan keseluruhan data yang benar atau positif dengan rumus 
Recall = (TP) / (TP + FN).

Alasan accuracy F1-score dipilih adalah karena pada kasus ATIS sangat menginginkan terjadinya precision yaitu True positif (prediksi class yang tepat berdasarkan kalimat yang tepat) dan recall yaitu tidak menginginkan adanya False positif (prediksi class yang salah berdasarkan kalimat yang tepat) 

Code yang digunakan sebagai berikut.

```py
from sklearn.metrics import classification_report

target_names = ['atis_flight','atis_flight_time','atis_airfare','atis_aircraft','atis_ground_service','atis_airline','atis_abbreviation','atis_quantity']
print(classification_report(data_test['label'], prediction_linear, target_names=target_names))

```


