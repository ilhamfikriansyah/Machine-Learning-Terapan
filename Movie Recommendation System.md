# Laporan Proyek Machine Learning - Ilham Fikriansyah

## Project Overview
Proyek yang dirancang kali ini merupakan proyek membuat sistem rekomendasi judul film berdasarkan genre menggunakan data dari metadata movies yang didownload dari kaggle. Permasalahan yang dapat diselesaikan dalam sistem rekomendasi ini adalah pelanggan dapat dengan mudah mendapatkan rekomendasi film berdasarkan genre dari judul film yang dicari atau yang telah diinput sebelumnya. Metode sistem rekomendasi judul film berdasarkan genre ini menggunakan metode _content-based filtering._

## Business Understanding
Proyek yang akan dijalankan adalah sistem rekomendasi yang dapat memprediksi judul film yang mungkin pelanggan suka berdasarkan genre dari judul film yang diinput metode yang akan digunakan yaitu teknik _content-based filtering_ dengan _simlarty measure_ yang digunakan adalah _Cosine Similarity_, dengan alur sebagai berikut.

### Problem Statements
Bagaimana memberikan rekomendasi judul film berdasarkan genre pada setiap judul film yang pelanggan input sehingga dapat memberikan referensi yang sesuai pelanggan inginkan?

### Goals
Membuat pelanggan lebih mudah menemukan judul film yang tepat dengan bantuan sistem rekomendasi judul film berdasarkan genre yang dibuat.

### Solution approach
Metode yang digunakan adalah _Content Based Filtering._ _Content Based Filtering_ adalah rekomendasi berbasis konten yang merekomendasikan item yang memiliki kemiripan dengan item yang disukai/di*input* pengguna sebelumnya.

_Content-based filtering_ mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Metode ini bekerja dengan menyarankan item serupa yang pernah disukai sebelumnya atau sedang dilihat sekarang kepada pengguna berdasrakan kategori tertentu dari item yang dinilai oleh pengguna.

Kelebihan dan Kekurangan dari _Content-based Filtering_
1. Kelebihan
    - _User Independence_ \
    Tidak bergantung kepada user lain dalam memberikan rekomendasi yang ada.
    - _New Item_ \
    Mampu merekomendasikan item yang belum dinilai oleh setiap pengguna.
2. Kekurangan
    - _New User_ \
    Sistem tidak dapat memberikan rekomendasi yang dapat diandalkan pada pengguna baru, karena membutuhkan penelusuran terlebih dahulu pada preferensi pengguna.

## Data Understanding
Dataset yang digunakan adalah [The Movie Dataset | Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset?select=movies_metadata.csv) dataset ini merupakan dataset mengenai film yang dirilis pada atau sebelum Juli 2017 yang berisi lebih dari 45.000 film. 26 juta peringkat dari lebih dari 270.000 pengguna. Namun data yang digunakan pada proyek ini kurang lebih hanya 5000 data dikarenakan spesifikasi laptop yang digunakan tidak kompatibel untuk data yang terlalu banyak.

Di dalam The Movie Dataset datasets ini berisi data Movies Meta Data, credits, links, ratings, dan keywords dengan kolom antara lain sebagai berikut.
- adult : mengkategorikan jenis film dewasa atau tidak
- budget : berisi budget yang digunakan dalam pembuatan film
- genres : berisi kategori genre film
- id : berisi id
- imbd_id : berisi id film
- original_language : berisi bahas yang digunakan
- original_title : berisi judul film
- popularity : berisi popularitas film
- production_companies : berisi produksi film
- ratings : berisi ratings fim
- release_date : berisi waktu rilis film


Variabel-variabel The Movie Dataset yang digunakan adalah sebagai berikut.
- Original_title : Berisi judul asli dari film
- genres : berisi satu atau lebih genre dari setiap judul film 

## Data Preparation
Teknik yang digunakan pada tahapan Data Preparation adalah vektorisasi fungsi CountVectorizer dari library scikit-learn. CountVectorizer digunakan untuk mengubah teks yang diberikan menjadi vektor berdasarkan frekuensi (jumlah) setiap kata yang muncul di seluruh teks. 

CountVectorizer membuat matriks di mana setiap kata unik diwakili oleh kolom matriks, dan setiap sampel teks dari dokumen adalah baris dalam matriks. Nilai setiap sel tidak lain adalah jumlah kata dalam sampel teks tertentu.

Pada proses vektorisasi ini, digunakan metode sebagai berikut. 
1. fit metode berfungsi untuk melakukan perhitungan idf pada data
2. get_feature berfungsi untuk melakukan mapping array dari fitur index integer ke fitur nama
3. fit_transform berfungsi untuk mempelajari kosa kata dan Inverse Document Frequency (IDF) dengan memberikan nilai return berupa *document-term matrix*
4. todense berfungsi untuk mengubah vektor tf-idf dalam bentuk matriks

## Modeling
Model machine learning yang digunakan pada sistem rekomendasi ini adalah model _content-based filtering_ dengan _simlarty measure_ yang digunakan adalah _Cosine Similarity_.

Model _content-based filtering_ ini bekerja dengan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Metode ini bekerja dengan menyarankan item serupa yang pernah disukai sebelumnya atau sedang dilihat sekarang kepada pengguna berdasrakan kategori tertentu dari item yang dinilai oleh pengguna dengan menggunakan _similarity_ tertentu.

Sedangkan _cosine similarity_ adalah salah satu teknik mengukur kesamaan yang bekerja dengan mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama dengan menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai _cosine similarity_.

Dalam pemanggilan rekomendasi judul film digunakan function yang dibuat dengan code sebagai berikut.
```py
def get_recommendations(judul, cosine_sim = cosine_sim,items=data[['judul','genre']]):
    # Mengambil indeks dari judul film yang telah didefinisikan sebelumnnya
    idx = indices[judul]
    
    # Mengambil skor kemiripan dengan semua judul film 
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Mengurutkan film berdasarkan skor kemiripan
    sim_scores = sorted(sim_scores, key = lambda x : x[1], reverse = True)
    
    # Mengambil 10 skor kemiripan dari 1-10 karena urutan 0 memberikan indeks yang sama dengan judul film yang diinput
    sim_scores = sim_scores[1:11]
    
    # Mengambil judul film dari skor kemiripan
    movie_indices = [i[0] for i in sim_scores]
    
    # Mengembalikan 10 rekomendasi judul film dari kemiripan skor yang telah diurutkan dan menampilkan genre dari 10 rekomendasi film tersebut
    return pd.DataFrame(data['judul'][movie_indices]).merge(items)
```

Tahapan yang dilakukan pada fungsi tersebut sebagai berikut.
1. Mengambil indeks dari judul film yang telah didefinisikan sebelumnnya
2. Mengambil skor kemiripan dengan semua film
3. Mengurutkan film berdasarkan skor kemiripan
4. Mengambil 10 skor kemiripan dari 1-10 karena urutan 0 memberikan indeks yang sama dengan judul film yang diinput
5. Mengambil judul film dari skor kemiripan
6. Mengembalikan 10 rekomendasi judul film dari kemiripan skor yang telah diurutkan dan menampilkan genre dari 10 rekomendasi film tersebut

Berikut _top_-10 _recommemdation_ berdasarkan genre dari judul film "*The American President*"
    
        judul                       genre
        The American President	    [Comedy, Drama, Romance]

Rekomendasi film

        judul	                    genre
    0	Nueba Yol	                [Comedy, Drama, Romance]
    1	飲食男女	                [Comedy, Drama, Romance]
    2	Only You	                [Comedy, Drama, Romance]
    3	Muriel's Wedding	        [Drama, Comedy, Romance]
    4	The Favor	                [Drama, Comedy, Romance]
    5	The Inkwell	                [Comedy, Drama, Romance]
    6	Meet John Doe	            [Drama, Comedy, Romance]
    7	The Pompatus of Love	    [Comedy, Romance, Drama]
    8	Jerry Maguire	            [Comedy, Drama, Romance]
    9	Roseanna's Grave	        [Comedy, Romance, Drama]

Dari hasil yang diberikan di atas berdasarkan judul film "*The American President*" dengan genre [Comedy, Drama, Romance] di dapatkan 10 rekomendasi judul film dengan genre yang serupa ataupun mirip.

## Evaluation
*Metric* yang digunakan pada sistem rekomendasi judul film berdasarkan genre adalah *accuracy precision*. *Precision* adalah metrik yang membandingkan rasio prediksi benar atau positif dengan keseluruhan hasil yang diprediksi positif dengan rumus
        
    Precission = (TP) / (TP+FP).
        
    keterangan:
    TP = True Positif (prediksi positif dan hal tersebut benar)
    FP = False Negatif (prediksi positif dan hal tersebut salah)

Alasan _accuracy Precision_ dipilih adalah karena metrik ini dapat membandingkan rasio prediksi benar atau positif dengan keseluruhan hasil yang diprediksi positif. Dalam hal ini adalah rasio item yang direkomendasikan memiliki genre yang mirip atau serupa dibandingkan dengan genre dari judul film yang diinput.

_Code_ yang digunakan untuk melihat jumlah genre yang mirip atau serupa adalah sebagai berikut.

```p
value = pd.DataFrame(rekomendasi['genre'].value_counts().reset_index().values, columns = ['genre', 'count'])
value.head()
```

_output_

	    genre               	    count
    0	[Comedy, Drama, Romance]	5
    1	[Drama, Comedy, Romance]	3
    2	[Comedy, Romance, Drama]	2

Dari output tersebut dihitung _accuracy precision_ nya adalah
```py
#jumlah prediksi benar untuk genre yang mirip atau serupa
TP = 10

#jumlah prediksi salah untuk genre yang mirip atau serupa
FP = 0 

Precision = TP/(TP+FP)
print("{0:.0%}".format(Precision))
```
_output_

        100%

Kesimpulan dari output yang dihasilkan bahwa prediksi rekomendasi yang diberikan 100% presisi sesuai genre yang mirip ayau serupa dengan genre dari judul yang diinput.
