```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121
```

Kode yang Anda tuliskan adalah bagian dari sebuah pipeline pemrosesan dan pelatihan model deep learning menggunakan TensorFlow dan Keras untuk klasifikasi gambar. Berikut adalah penjelasan setiap barisnya:

1. import numpy as np:
   - Mengimpor library NumPy yang digunakan untuk operasi array dan matriks.
   
2. import pandas as pd:
   - Mengimpor library Pandas yang digunakan untuk manipulasi data dan analisis.

3. import matplotlib.pyplot as plt:
   - Mengimpor library Matplotlib yang digunakan untuk membuat plot dan visualisasi data.

4. %matplotlib inline:
   - Sebuah magic function dari Jupyter Notebook yang memungkinkan hasil plot Matplotlib ditampilkan langsung di notebook.

5. import seaborn as sns:
   - Mengimpor library Seaborn yang digunakan untuk visualisasi data statistik.

6. import cv2:
   - Mengimpor library OpenCV yang digunakan untuk pemrosesan gambar.

7. import os:
   - Mengimpor modul os yang menyediakan fungsi untuk berinteraksi dengan sistem operasi.

8. from tqdm import tqdm:
   - Mengimpor fungsi tqdm yang digunakan untuk membuat progress bar di loop.

9. from sklearn.metrics import confusion_matrix:
   - Mengimpor fungsi confusion_matrix dari scikit-learn yang digunakan untuk mengevaluasi hasil klasifikasi.

10. from sklearn.model_selection import train_test_split:
    - Mengimpor fungsi train_test_split dari scikit-learn yang digunakan untuk membagi dataset menjadi set pelatihan dan pengujian.

11. import tensorflow as tf:
    - Mengimpor library TensorFlow yang digunakan untuk membangun dan melatih model machine learning dan deep learning.

12. from tensorflow.keras.utils import to_categorical:
    - Mengimpor fungsi to_categorical dari Keras yang digunakan untuk mengonversi label kelas menjadi format one-hot encoding.

13. from tensorflow.keras.models import Model, Sequential, load_model:
    - Mengimpor class Model, Sequential, dan fungsi load_model dari Keras yang digunakan untuk membangun dan memuat model.

14. from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D:
    - Mengimpor berbagai lapisan dari Keras yang digunakan untuk membangun arsitektur neural network.

15. from tensorflow.keras.optimizers import Adam:
    - Mengimpor optimizer Adam dari Keras yang digunakan untuk mengoptimalkan proses pelatihan model.

16. from tensorflow.keras.preprocessing.image import ImageDataGenerator:
    - Mengimpor class ImageDataGenerator dari Keras yang digunakan untuk augmentasi gambar secara real-time selama pelatihan.

17. from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau:
    - Mengimpor callback ModelCheckpoint dan ReduceLROnPlateau dari Keras yang digunakan untuk menyimpan model terbaik dan menyesuaikan learning rate selama pelatihan.

18. from tensorflow.keras.applications import DenseNet121:
    - Mengimpor arsitektur DenseNet121 dari Keras yang merupakan model deep learning pralatih untuk klasifikasi gambar.

Kode ini mempersiapkan berbagai library dan modul yang diperlukan untuk memproses dataset gambar, membangun model neural network, melatih model tersebut, dan mengevaluasi hasilnya.

```
disease_types=['COVID', 'non-COVID']
data_dir = '/Users/asus/covid-sars/'
train_dir = os.path.join(data_dir)

```
Kode ini menyiapkan beberapa variabel yang digunakan untuk mengakses dataset dan menentukan jenis penyakit yang akan diklasifikasikan. Berikut penjelasannya:

1. disease_types=['COVID', 'non-COVID']:
   - Mendefinisikan list disease_types yang berisi dua jenis penyakit yang akan diklasifikasikan, yaitu 'COVID' dan 'non-COVID'.

2. data_dir = '/Users/asus/covid-sars/':
   - Mendefinisikan variabel data_dir yang menyimpan path ke direktori utama tempat dataset disimpan. Path ini menunjukkan lokasi direktori pada sistem file lokal Anda.

3. train_dir = os.path.join(data_dir):
   - Menggunakan fungsi os.path.join untuk menggabungkan data_dir dengan direktori tambahan (jika ada). Dalam kasus ini, data_dir dan train_dir akan sama karena tidak ada direktori tambahan yang ditambahkan.

Berikut adalah kode lengkap dengan beberapa komentar tambahan:

python
# Definisikan jenis penyakit yang akan diklasifikasikan
disease_types = ['COVID', 'non-COVID']

# Tentukan direktori utama tempat dataset disimpan
data_dir = '/Users/asus/covid-sars/'

# Tentukan direktori pelatihan dengan menggabungkan path data_dir
train_dir = os.path.join(data_dir)


Dengan penyiapan ini, Anda dapat menggunakan train_dir untuk mengakses dataset Anda selama proses pemuatan dan pemrosesan data.

```
train_data = []
for defects_id, sp in enumerate(disease_types):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}/{}'.format(sp, file), defects_id, sp])
        
train = pd.DataFrame(train_data, columns=['File', 'DiseaseID','Disease Type'])
train.head()

```
Kode ini memuat data pelatihan dari direktori yang telah ditentukan dan menyimpannya dalam sebuah DataFrame. Berikut adalah penjelasan rinci dari setiap bagian kode:

1. train_data = []:
   - Menginisialisasi list kosong train_data untuk menyimpan informasi tentang setiap gambar dalam dataset.

2. for defects_id, sp in enumerate(disease_types)::
   - Mengiterasi melalui list disease_types dengan enumerate, sehingga defects_id akan berisi indeks (0 untuk 'COVID', 1 untuk 'non-COVID') dan sp akan berisi nama penyakit ('COVID' atau 'non-COVID').

3. for file in os.listdir(os.path.join(train_dir, sp))::
   - Mengiterasi melalui semua file dalam subdirektori sp di dalam train_dir. os.listdir mengembalikan daftar semua file dalam direktori yang ditentukan.

4. train_data.append(['{}/{}'.format(sp, file), defects_id, sp]):
   - Menambahkan informasi tentang setiap file gambar ke dalam list train_data. Informasi yang ditambahkan mencakup:
     - Path relatif ke file gambar ('{}/{}'.format(sp, file)).
     - ID penyakit (defects_id).
     - Nama penyakit (sp).

5. train = pd.DataFrame(train_data, columns=['File', 'DiseaseID', 'Disease Type']):
   - Mengonversi list train_data menjadi sebuah DataFrame Pandas dengan kolom 'File', 'DiseaseID', dan 'Disease Type'.

6. train.head():
   - Menampilkan lima baris pertama dari DataFrame train untuk memverifikasi bahwa data telah dimuat dengan benar.

Berikut adalah kode lengkap dengan beberapa komentar tambahan:

python
# Inisialisasi list kosong untuk menyimpan data pelatihan
train_data = []

# Iterasi melalui jenis penyakit
for defects_id, sp in enumerate(disease_types):
    # Iterasi melalui semua file dalam subdirektori
    for file in os.listdir(os.path.join(train_dir, sp)):
        # Tambahkan informasi tentang setiap file gambar ke dalam list train_data
        train_data.append(['{}/{}'.format(sp, file), defects_id, sp])
        
# Konversi list train_data menjadi DataFrame Pandas
train = pd.DataFrame(train_data, columns=['File', 'DiseaseID', 'Disease Type'])

# Tampilkan lima baris pertama dari DataFrame train
train.head()


Hasil dari kode ini adalah DataFrame train yang berisi informasi tentang setiap file gambar dalam dataset, termasuk path relatif ke file, ID penyakit, dan nama penyakit. DataFrame ini dapat digunakan untuk memuat dan memproses gambar selama pelatihan model.

```
SEED = 42
train = train.sample(frac=1, random_state=SEED) 
train.index = np.arange(len(train)) # Reset indices
train.head()

```
Kode ini mengacak urutan baris dalam DataFrame train untuk memastikan bahwa data pelatihan tidak terurut berdasarkan kelas penyakit. Hal ini penting untuk menghindari bias selama pelatihan model. Berikut adalah penjelasan rinci dari setiap bagian kode:

1. SEED = 42:
   - Mendefinisikan variabel SEED yang digunakan sebagai seed untuk fungsi pengacakan. Seed ini memastikan bahwa pengacakan dapat direproduksi (dengan seed yang sama, hasil pengacakan akan selalu sama).

2. train = train.sample(frac=1, random_state=SEED):
   - Menggunakan metode sample dari DataFrame Pandas untuk mengacak urutan baris dalam DataFrame train.
   - frac=1 berarti semua baris akan diambil dan diacak.
   - random_state=SEED memastikan bahwa pengacakan dapat direproduksi.

3. train.index = np.arange(len(train)):
   - Mengatur ulang indeks DataFrame train agar mulai dari 0 hingga jumlah baris dikurangi satu. Ini dilakukan dengan membuat array dari 0 hingga panjang DataFrame train menggunakan np.arange(len(train)).

4. train.head():
   - Menampilkan lima baris pertama dari DataFrame train yang telah diacak untuk memverifikasi bahwa data telah diacak dan indeks telah diatur ulang.

Berikut adalah kode lengkap dengan beberapa komentar tambahan:

python
# Definisikan seed untuk pengacakan
SEED = 42

# Acak urutan baris dalam DataFrame train
train = train.sample(frac=1, random_state=SEED)

# Atur ulang indeks DataFrame train agar mulai dari 0 hingga jumlah baris dikurangi satu
train.index = np.arange(len(train))

# Tampilkan lima baris pertama dari DataFrame train yang telah diacak
train.head()


Hasil dari kode ini adalah DataFrame train yang telah diacak urutannya, dengan indeks yang telah diatur ulang, siap untuk digunakan dalam proses pelatihan model tanpa bias urutan data.


```
plt.hist(train['DiseaseID'])
plt.title('Frequency Histogram of Species')
plt.figure(figsize=(12, 12))
plt.show()

```

Kode ini membuat histogram frekuensi dari kolom 'DiseaseID' dalam DataFrame train untuk memvisualisasikan distribusi jumlah gambar per kategori penyakit. Berikut adalah penjelasan rinci dari setiap bagian kode:

1. plt.hist(train['DiseaseID']):
   - Membuat histogram dari kolom 'DiseaseID' dalam DataFrame train.
   - Histogram menunjukkan frekuensi (jumlah) dari setiap nilai unik dalam kolom 'DiseaseID'.

2. plt.title('Frequency Histogram of Species'):
   - Menambahkan judul pada histogram.

3. plt.figure(figsize=(12, 12)):
   - Mengatur ukuran dari figure yang akan ditampilkan. Namun, ini tidak akan berdampak pada figure yang sudah ditampilkan oleh plt.hist. Sebaiknya dipindahkan ke sebelum plt.hist.

4. plt.show():
   - Menampilkan plot yang telah dibuat.

Berikut adalah kode lengkap dengan beberapa komentar tambahan serta perbaikan dalam urutan pemanggilan fungsi plt.figure:

python
# Mengatur ukuran figure terlebih dahulu
plt.figure(figsize=(12, 12))

# Membuat histogram dari kolom 'DiseaseID' dalam DataFrame train
plt.hist(train['DiseaseID'])

# Menambahkan judul pada histogram
plt.title('Frequency Histogram of Species')

# Menampilkan plot
plt.show()


Hasil dari kode ini adalah sebuah histogram yang menunjukkan distribusi jumlah gambar dalam setiap kategori penyakit ('COVID' dan 'non-COVID'), yang memungkinkan Anda untuk melihat apakah dataset Anda seimbang atau tidak.


```
def plot_defects(defect_types, rows, cols):
    fig, ax = plt.subplots(rows, cols, figsize=(12, 12))
    defect_files = train['File'][train['Disease Type'] == defect_types].values
    n = 0
    for i in range(rows):
        for j in range(cols):
            image_path = os.path.join(data_dir, defect_files[n])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(cv2.imread(image_path))
            n += 1
# Displays first n images of class from training set
plot_defects('COVID', 5, 5)
```

Fungsi plot_defects ini digunakan untuk memplot beberapa contoh gambar dari kelas tertentu (dalam hal ini, 'COVID') dari dataset pelatihan. Berikut adalah penjelasan dari kode tersebut:

1. def plot_defects(defect_types, rows, cols):
   - Mendefinisikan fungsi plot_defects dengan tiga parameter: defect_types untuk jenis penyakit yang akan diplot, rows untuk jumlah baris plot, dan cols untuk jumlah kolom plot.

2. fig, ax = plt.subplots(rows, cols, figsize=(12, 12))
   - Membuat sebuah figure dan array dari axes dengan ukuran sesuai yang telah ditentukan (rows dan cols), dengan ukuran keseluruhan figure (figsize) adalah 12x12.

3. defect_files = train['File'][train['Disease Type'] == defect_types].values
   - Memilih file-file gambar dari DataFrame train yang memiliki 'Disease Type' sesuai dengan defect_types (dalam kasus ini, 'COVID').
   - .values digunakan untuk mengambil nilai-nilai sebagai array NumPy.

4. n = 0
   - Inisialisasi variabel n sebagai indeks untuk mengakses defect_files.

5. Loop for i in range(rows) dan for j in range(cols):
   - Digunakan untuk mengiterasi melalui setiap sel di grid plot yang telah dibuat.

6. image_path = os.path.join(data_dir, defect_files[n])
   - Mendapatkan path lengkap ke file gambar yang akan diplot dengan menggabungkan data_dir dengan defect_files[n].

7. ax[i, j].set_xticks([]) dan ax[i, j].set_yticks([])
   - Menghilangkan tanda sumbu x dan y pada setiap subplot.

8. ax[i, j].imshow(cv2.imread(image_path))
   - Membaca gambar dari image_path menggunakan OpenCV (cv2.imread()) dan menampilkan gambar tersebut pada subplot yang sesuai.

9. n += 1
   - Memperbarui indeks n untuk memilih gambar berikutnya dari defect_files.

Fungsi ini akan menampilkan grid 5x5 (atau sesuai dengan nilai rows dan cols yang Anda berikan) dari gambar COVID dari dataset pelatihan.

Pastikan bahwa variabel data_dir dan train sudah didefinisikan dengan benar sebelum memanggil fungsi ini.


```
def plot_defects(defect_types, rows, cols):
    fig, ax = plt.subplots(rows, cols, figsize=(12, 12))
    defect_files = train['File'][train['Disease Type'] == defect_types].values
    n = 0
    for i in range(rows):
        for j in range(cols):
            image_path = os.path.join(data_dir, defect_files[n])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(cv2.imread(image_path))
            n += 1
# Displays first n images of class from training set
plot_defects('non-COVID', 5, 5)
```

Fungsi plot_defects yang Anda definisikan digunakan untuk memplot beberapa contoh gambar dari kelas tertentu (dalam kasus ini, 'non-COVID') dari dataset pelatihan. Berikut adalah cara kerjanya:

1. def plot_defects(defect_types, rows, cols):
   - Mendefinisikan fungsi plot_defects dengan tiga parameter: defect_types untuk jenis penyakit yang akan diplot, rows untuk jumlah baris plot, dan cols untuk jumlah kolom plot.

2. fig, ax = plt.subplots(rows, cols, figsize=(12, 12))
   - Membuat sebuah figure dan array dari axes dengan ukuran sesuai yang telah ditentukan (rows dan cols), dengan ukuran keseluruhan figure (figsize) adalah 12x12.

3. defect_files = train['File'][train['Disease Type'] == defect_types].values
   - Memilih file-file gambar dari DataFrame train yang memiliki 'Disease Type' sesuai dengan defect_types (dalam kasus ini, 'non-COVID').
   - .values digunakan untuk mengambil nilai-nilai sebagai array NumPy.

4. n = 0
   - Inisialisasi variabel n sebagai indeks untuk mengakses defect_files.

5. Loop for i in range(rows) dan for j in range(cols):
   - Digunakan untuk mengiterasi melalui setiap sel di grid plot yang telah dibuat.

6. image_path = os.path.join(data_dir, defect_files[n])
   - Mendapatkan path lengkap ke file gambar yang akan diplot dengan menggabungkan data_dir dengan defect_files[n].

7. ax[i, j].set_xticks([]) dan ax[i, j].set_yticks([])
   - Menghilangkan tanda sumbu x dan y pada setiap subplot.

8. ax[i, j].imshow(cv2.imread(image_path))
   - Membaca gambar dari image_path menggunakan OpenCV (cv2.imread()) dan menampilkan gambar tersebut pada subplot yang sesuai.

9. n += 1
   - Memperbarui indeks n untuk memilih gambar berikutnya dari defect_files.

Fungsi ini akan menampilkan grid 5x5 (atau sesuai dengan nilai rows dan cols yang Anda berikan) dari gambar 'non-COVID' dari dataset pelatihan.

Pastikan bahwa variabel data_dir dan train sudah didefinisikan dengan benar sebelum memanggil fungsi ini.


```
IMAGE_SIZE = 64
def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) # Loading a color image is the default flag
# Resize image to target size
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)
```

Fungsi-fungsi read_image dan resize_image yang Anda definisikan berfungsi untuk membaca dan mengubah ukuran gambar menggunakan library OpenCV (cv2). Berikut adalah penjelasan singkat dari masing-masing fungsi:

1. read_image(filepath):
   - Fungsi ini mengambil path relatif filepath gambar, bergabung dengan data_dir, dan menggunakan cv2.imread untuk membaca gambar dalam mode default (gambar berwarna).
   - *Return*: Mengembalikan gambar yang dibaca dalam bentuk array NumPy.

2. resize_image(image, image_size):
   - Fungsi ini mengambil gambar image dan mengubah ukurannya menjadi image_size menggunakan cv2.resize.
   - image.copy() digunakan untuk membuat salinan gambar agar tidak merusak gambar asli.
   - interpolation=cv2.INTER_AREA digunakan untuk menentukan metode interpolasi yang digunakan saat mengubah ukuran gambar.
   - *Return*: Mengembalikan gambar yang telah diubah ukurannya dalam bentuk array NumPy.

Berikut adalah contoh implementasi dari kedua fungsi tersebut:

python
import cv2
import os

# Definisi ukuran gambar target
IMAGE_SIZE = 64

# Fungsi untuk membaca gambar
def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath))

# Fungsi untuk mengubah ukuran gambar
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)


Anda dapat menggunakan fungsi read_image untuk membaca gambar dari path tertentu dan resize_image untuk mengubah ukuran gambar tersebut sesuai dengan ukuran yang diinginkan (IMAGE_SIZE). Pastikan variabel data_dir sudah didefinisikan sebelum menggunakan fungsi read_image.


```
X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        X_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
# Normalize the data
X_Train = X_train / 255.
print('Train Shape: {}'.format(X_Train.shape))
```

Kode ini melakukan preprocessing pada dataset gambar untuk digunakan dalam pelatihan model machine learning atau deep learning. Berikut adalah penjelasan dari setiap bagian kode:

1. X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3)):
   - Membuat array X_train yang berukuran (jumlah_data, IMAGE_SIZE, IMAGE_SIZE, 3).
   - train.shape[0] memberikan jumlah baris dalam DataFrame train, yang berarti jumlah data pelatihan.
   - IMAGE_SIZE adalah ukuran yang telah ditentukan untuk gambar (64x64 piksel).
   - 3 adalah jumlah saluran warna (RGB) untuk gambar berwarna.

2. for i, file in tqdm(enumerate(train['File'].values))::
   - Melakukan iterasi melalui setiap file gambar dalam kolom 'File' dari DataFrame train.
   - enumerate(train['File'].values) memberikan indeks (i) dan nilai (file) dari setiap file gambar.

3. image = read_image(file):
   - Menggunakan fungsi read_image untuk membaca gambar dari file.

4. if image is not None::
   - Memeriksa apakah gambar berhasil dibaca (image bukan None).

5. X_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE)):
   - Menggunakan fungsi resize_image untuk mengubah ukuran gambar menjadi (IMAGE_SIZE, IMAGE_SIZE) dan menyimpannya di dalam array X_train pada indeks i.

6. X_Train = X_train / 255.:
   - Melakukan normalisasi data dengan membagi setiap nilai piksel dalam X_train dengan 255.
   - Normalisasi umumnya dilakukan untuk mengubah rentang nilai piksel menjadi 0-1, yang membantu dalam pelatihan model.

7. print('Train Shape: {}'.format(X_Train.shape)):
   - Mencetak bentuk dari X_Train, yang menunjukkan jumlah data pelatihan dan ukuran gambar setelah preprocessing.

Berikut adalah kode lengkap dengan beberapa komentar tambahan:

python
# Inisialisasi array X_train dengan ukuran yang sesuai
X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))

# Iterasi melalui setiap file gambar dalam DataFrame train
for i, file in tqdm(enumerate(train['File'].values)):
    # Membaca gambar dari file menggunakan read_image
    image = read_image(file)
    # Memeriksa apakah gambar berhasil dibaca
    if image is not None:
        # Mengubah ukuran gambar dan menyimpannya ke dalam X_train
        X_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))

# Normalisasi data dengan membagi setiap nilai piksel dengan 255
X_Train = X_train / 255.

# Mencetak bentuk dari X_Train setelah preprocessing
print('Train Shape: {}'.format(X_Train.shape))


Hasil dari kode ini adalah X_Train, yang berisi gambar-gambar yang telah diubah ukurannya dan dinormalisasi, siap untuk digunakan dalam proses pelatihan model machine learning atau deep learning. Pastikan bahwa variabel data_dir, train, read_image, dan resize_image sudah didefinisikan dan berfungsi dengan benar sebelum menjalankan kode ini.


```
Y_train = train['DiseaseID'].values
Y_train = to_categorical(Y_train, num_classes=2)
```

Kode ini menghasilkan variabel Y_train yang berisi label kelas dalam bentuk one-hot encoding untuk digunakan dalam pelatihan model klasifikasi. Berikut adalah penjelasan dari setiap baris kode:

1. Y_train = train['DiseaseID'].values:
   - Mengambil nilai dari kolom 'DiseaseID' dari DataFrame train dan menyimpannya dalam array Y_train.
   - Ini berisi label kelas untuk setiap gambar dalam dataset pelatihan.

2. Y_train = to_categorical(Y_train, num_classes=2):
   - Menggunakan fungsi to_categorical dari Keras untuk mengonversi label kelas (Y_train) menjadi format one-hot encoding.
   - num_classes=2 menentukan jumlah kelas yang akan diencode. Dalam hal ini, terdapat dua kelas: 'COVID' dan 'non-COVID'.

Jadi, jika Y_train awalnya berisi label kelas seperti [0, 1, 0, 1, ...], setelah pengonversian menjadi one-hot encoding, Y_train akan menjadi array dua dimensi di mana setiap baris mewakili satu data dengan nilai 1 yang menunjukkan kelas yang benar dan nilai 0 untuk kelas lainnya.

Berikut adalah contoh implementasi lengkap dari kedua langkah ini:

python
import numpy as np
from tensorflow.keras.utils import to_categorical

# Mengambil nilai dari kolom 'DiseaseID' sebagai Y_train
Y_train = train['DiseaseID'].values

# Mengonversi Y_train menjadi format one-hot encoding dengan num_classes=2
Y_train = to_categorical(Y_train, num_classes=2)


Setelah menjalankan kode ini, Y_train siap digunakan sebagai target output (label) dalam proses pelatihan model machine learning atau deep learning yang bertujuan untuk klasifikasi antara 'COVID' dan 'non-COVID'. Pastikan variabel train telah didefinisikan dan berisi data yang benar sebelum menjalankan kode ini.


```
BATCH_SIZE = 64

# Split the train and validation sets 
X_train, X_val, Y_train, Y_val = train_test_split(X_Train, Y_train, test_size=0.2, random_state=SEED)
```


Kode ini menggunakan fungsi train_test_split dari sklearn.model_selection untuk membagi data menjadi set pelatihan dan validasi. Berikut adalah penjelasan singkat dari setiap baris kode:

1. BATCH_SIZE = 64:
   - Mendefinisikan ukuran batch yang akan digunakan selama pelatihan model. Ini menentukan jumlah sampel yang akan digunakan dalam satu iterasi pelatihan.

2. X_train, X_val, Y_train, Y_val = train_test_split(X_Train, Y_train, test_size=0.2, random_state=SEED):
   - Memanggil fungsi train_test_split untuk membagi data.
   - X_Train adalah dataset gambar yang sudah di-preprocess dan dinormalisasi.
   - Y_train adalah label kelas dalam format one-hot encoding yang sudah disiapkan sebelumnya.
   - test_size=0.2 menentukan proporsi data yang akan dialokasikan untuk set validasi. Dalam hal ini, 20% dari data akan digunakan sebagai validasi.
   - random_state=SEED digunakan untuk mengatur seed agar pemisahan data menjadi konsisten dan dapat direproduksi.

Setelah kode dijalankan, variabel X_train, X_val, Y_train, dan Y_val akan berisi:
- X_train: Dataset gambar untuk pelatihan.
- Y_train: Label kelas dalam format one-hot encoding untuk pelatihan.
- X_val: Dataset gambar untuk validasi.
- Y_val: Label kelas dalam format one-hot encoding untuk validasi.

Berikut adalah contoh implementasi lengkap dari kedua langkah ini:

python
from sklearn.model_selection import train_test_split

# Definisikan BATCH_SIZE
BATCH_SIZE = 64

# Bagi dataset menjadi set pelatihan dan validasi
X_train, X_val, Y_train, Y_val = train_test_split(X_Train, Y_train, test_size=0.2, random_state=SEED)


Pastikan variabel X_Train dan Y_train sudah didefinisikan dan berisi data yang benar sebelum menjalankan kode ini.


```
fig, ax = plt.subplots(1, 3, figsize=(15, 15))
for i in range(3):
    ax[i].set_axis_off()
    ax[i].imshow(X_train[i])
    ax[i].set_title(disease_types[np.argmax(Y_train[i])])
```


Kode ini bertujuan untuk menampilkan tiga contoh gambar dari dataset pelatihan beserta labelnya dalam bentuk plot subplot menggunakan Matplotlib. Berikut adalah penjelasan dari setiap baris kode:

1. fig, ax = plt.subplots(1, 3, figsize=(15, 15)):
   - Membuat sebuah figure dengan satu baris (1) dan tiga kolom (3).
   - figsize=(15, 15) menentukan ukuran keseluruhan figure yang akan ditampilkan.

2. for i in range(3)::
   - Melakukan iterasi tiga kali (karena ingin menampilkan tiga gambar).

3. ax[i].set_axis_off():
   - Menonaktifkan sumbu (axis) pada subplot ke-i agar tidak terlihat.

4. ax[i].imshow(X_train[i]):
   - Menampilkan gambar ke-i dari X_train pada subplot ke-i menggunakan imshow dari Matplotlib.

5. ax[i].set_title(disease_types[np.argmax(Y_train[i])]):
   - Menetapkan judul pada subplot ke-i yang merupakan label kelas dari gambar tersebut.
   - np.argmax(Y_train[i]) digunakan untuk mendapatkan indeks kelas dengan nilai maksimum dari array one-hot encoded Y_train[i].
   - disease_types[np.argmax(Y_train[i])] digunakan untuk mengambil nama penyakit dari disease_types berdasarkan indeks kelas yang ditemukan sebelumnya.

Berikut adalah contoh implementasi lengkap dari kode ini:

python
import matplotlib.pyplot as plt

# Membuat figure dan axes
fig, ax = plt.subplots(1, 3, figsize=(15, 15))

# Iterasi untuk menampilkan tiga gambar
for i in range(3):
    # Menonaktifkan sumbu
    ax[i].set_axis_off()
    # Menampilkan gambar dari X_train
    ax[i].imshow(X_train[i])
    # Menetapkan judul berdasarkan kelas
    ax[i].set_title(disease_types[np.argmax(Y_train[i])])

# Menampilkan plot
plt.show()


Hasil dari kode ini adalah tiga subplot yang menampilkan gambar-gambar dari dataset pelatihan beserta label kelasnya. Setiap gambar diikuti oleh nama penyakit yang sesuai dengan label kelasnya. Pastikan bahwa X_train, Y_train, dan disease_types telah didefinisikan dengan benar sebelum menjalankan kode ini.


```
EPOCHS = 50
SIZE=64
N_ch=3
```

Variabel yang Anda definisikan, yaitu EPOCHS, SIZE, dan N_ch, biasanya digunakan dalam konteks pelatihan model deep learning untuk mengontrol jumlah epochs, ukuran gambar, dan jumlah saluran warna (channels) gambar. Berikut adalah penjelasan singkat dari setiap variabel:

1. EPOCHS = 50:
   - Variabel ini menentukan jumlah iterasi (epochs) yang akan dilakukan selama pelatihan model. Setiap epoch mencakup satu iterasi dari seluruh dataset pelatihan yang digunakan untuk menyesuaikan bobot model.

2. SIZE = 64:
   - Variabel ini menentukan ukuran yang diinginkan untuk gambar. Dalam konteks ini, SIZE sering kali mengacu pada tinggi atau lebar gambar dalam piksel setelah proses resizing.

3. N_ch = 3:
   - Variabel ini menentukan jumlah saluran warna (channels) dalam gambar. Misalnya, untuk gambar RGB (Red, Green, Blue), N_ch akan bernilai 3 karena terdapat tiga saluran warna.

Ketiga variabel ini sering digunakan dalam berbagai tahap pemrosesan dan pelatihan model deep learning. Misalnya, saat memuat dan memproses dataset gambar, Anda akan memperhatikan SIZE untuk mengatur ukuran gambar, dan N_ch untuk menyesuaikan input layer dari model Anda agar sesuai dengan saluran warna gambar yang Anda gunakan.

Jika Anda ingin menggunakan variabel-variabel ini dalam konteks implementasi lebih lanjut, pastikan untuk menyesuaikan parameter-parameter yang sesuai dalam setiap fungsi atau proses yang membutuhkan informasi tentang ukuran gambar atau jumlah epochs yang diinginkan.


```
def build_densenet():
    densenet = DenseNet121(weights='imagenet', include_top=False)

    input = Input(shape=(SIZE, SIZE, N_ch))
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    output = Dense(2,activation = 'softmax', name='root')(x)
 

    model = Model(inputs=input, outputs=output)
    
    optimizer = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    return model
```

Fungsi build_densenet yang Anda definisikan bertujuan untuk membangun dan mengompilasi model neural network menggunakan arsitektur DenseNet121 sebagai bagian utama dari model. Berikut adalah penjelasan dari setiap bagian kode:

1. densenet = DenseNet121(weights='imagenet', include_top=False):
   - Memuat pre-trained model DenseNet121 dari Keras dengan menggunakan bobot (weights) yang telah dilatih pada dataset ImageNet.
   - include_top=False menghilangkan layer fully connected teratas (top layer) yang biasanya digunakan untuk klasifikasi 1000 kelas dari ImageNet.

2. input = Input(shape=(SIZE, SIZE, N_ch)):
   - Mendefinisikan layer input untuk model dengan ukuran (SIZE, SIZE, N_ch).
   - SIZE dan N_ch didasarkan pada variabel yang telah Anda definisikan sebelumnya.

3. x = Conv2D(3, (3, 3), padding='same')(input):
   - Layer Conv2D pertama dengan 3 filter, ukuran kernel (3,3), dan padding 'same'.
   - Ini digunakan untuk mengubah saluran warna gambar menjadi tiga saluran, sesuai dengan N_ch, agar sesuai dengan masukan yang diharapkan oleh DenseNet121.

4. x = densenet(x):
   - Menyusun bagian DenseNet121 dari model di atas input x.

5. x = GlobalAveragePooling2D()(x):
   - Menggunakan Global Average Pooling untuk mengubah matriks fitur menjadi vektor fitur dengan rata-rata dari masing-masing saluran.
   - Ini membantu mengurangi dimensi fitur sebelum lapisan Dense.

6. x = BatchNormalization()(x):
   - Menambahkan layer Batch Normalization untuk mengurangi internal covariate shift dan meningkatkan stabilitas serta kecepatan konvergensi model.

7. x = Dropout(0.5)(x):
   - Menambahkan Dropout untuk mencegah overfitting dengan mengabaikan setengah dari unit selama pelatihan.

8. x = Dense(256, activation='relu')(x):
   - Menambahkan Dense layer dengan 256 unit dan fungsi aktivasi ReLU.

9. x = BatchNormalization()(x):
   - Lagi-lagi menambahkan Batch Normalization untuk stabilitas model.

10. x = Dropout(0.5)(x):
    - Lagi-lagi menambahkan Dropout untuk mencegah overfitting.

11. output = Dense(2,activation = 'softmax', name='root')(x):
    - Output layer dengan 2 unit dan fungsi aktivasi softmax untuk klasifikasi dua kelas ('COVID' dan 'non-COVID').

12. model = Model(inputs=input, outputs=output):
    - Membuat objek model dengan menyediakan input dan output.

13. optimizer = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0):
    - Menggunakan optimizer Adam dengan parameter-parameter yang telah ditentukan.

14. model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']):
    - Mengompilasi model dengan fungsi loss 'categorical_crossentropy' untuk klasifikasi multi-kelas dan metrics 'accuracy' untuk evaluasi performa.

15. model.summary():
    - Menampilkan ringkasan dari arsitektur model yang telah dibangun, termasuk jumlah parameter yang bisa di-train dan non-trainable.

16. return model:
    - Mengembalikan objek model yang telah dibangun dan dikompilasi.

Ini adalah sebuah contoh model yang menggunakan DenseNet121 sebagai bagian pengolah fitur (feature extractor) dan ditambahkan beberapa lapisan Dense, Batch Normalization, dan Dropout untuk pembelajaran dan generalisasi lebih lanjut. Pastikan Anda memiliki library yang diperlukan dan dependensi yang terpenuhi sebelum mencoba menjalankan atau melatih model ini.


```
model = build_densenet()
annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint('model.keras', verbose=1, save_best_only=True)
# Generates batches of image data with data augmentation
datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations
                        width_shift_range=0.2, # Range for random horizontal shifts
                        height_shift_range=0.2, # Range for random vertical shifts
                        zoom_range=0.2, # Range for random zoom
                        horizontal_flip=True, # Randomly flip inputs horizontally
                        vertical_flip=True) # Randomly flip inputs vertically

datagen.fit(X_train)
# Fits the model on batches with real-time data augmentation
hist = model.fit(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                 steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                 epochs=EPOCHS,
                 verbose=2,
                 callbacks=[annealer, checkpoint],
                 validation_data=(X_val, Y_val))
```

Kode ini melakukan beberapa langkah penting dalam proses pelatihan model menggunakan data augmentation dan callbacks di TensorFlow/Keras. Berikut adalah penjelasan dari setiap bagian kode:

1. model = build_densenet():
   - Memanggil fungsi build_densenet() untuk membuat dan mengompilasi model DenseNet121 yang telah Anda definisikan sebelumnya.

2. annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3):
   - Membuat objek callback ReduceLROnPlateau untuk mengurangi learning rate ketika tidak terjadi peningkatan val_accuracy dalam beberapa epoch (patience=5).

3. checkpoint = ModelCheckpoint('model.keras', verbose=1, save_best_only=True):
   - Membuat objek callback ModelCheckpoint untuk menyimpan model terbaik ke dalam file 'model.keras' berdasarkan val_accuracy.

4. datagen = ImageDataGenerator(rotation_range=360, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True):
   - Membuat objek ImageDataGenerator untuk melakukan augmentasi data secara real-time selama pelatihan.
   - rotation_range: Rentang derajat untuk rotasi gambar secara acak.
   - width_shift_range dan height_shift_range: Rentang untuk pergeseran horizontal dan vertikal secara acak.
   - zoom_range: Rentang untuk zoom secara acak.
   - horizontal_flip dan vertical_flip: Melempar gambar secara horizontal dan vertikal secara acak.

5. datagen.fit(X_train):
   - Menyesuaikan datagen dengan dataset pelatihan (X_train) untuk menghitung statistik yang diperlukan untuk augmentasi data.

6. hist = model.fit(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE), steps_per_epoch=X_train.shape[0] // BATCH_SIZE, epochs=EPOCHS, verbose=2, callbacks=[annealer, checkpoint], validation_data=(X_val, Y_val)):
   - Melatih model menggunakan augmented data generator (datagen.flow) untuk menghasilkan batch data augmented secara real-time.
   - steps_per_epoch=X_train.shape[0] // BATCH_SIZE menentukan jumlah batch yang akan digunakan dalam setiap epoch.
   - epochs=EPOCHS menentukan jumlah epoch pelatihan.
   - verbose=2 menampilkan detail pelatihan di setiap epoch.
   - callbacks=[annealer, checkpoint] menyertakan callback annealer dan checkpoint yang telah dibuat sebelumnya.
   - validation_data=(X_val, Y_val) menentukan data validasi untuk mengukur kinerja model selama pelatihan.

Kode ini akan menghasilkan hist, yang berisi riwayat pelatihan dari model, termasuk nilai loss dan metrik lainnya untuk setiap epoch. Pastikan bahwa semua variabel seperti BATCH_SIZE, EPOCHS, X_train, Y_train, X_val, dan Y_val telah didefinisikan dengan benar sebelum menjalankan kode ini.



```
final_loss, final_accuracy = model.evaluate(X_val, Y_val)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))
```

Kode ini digunakan untuk mengevaluasi model pada data validasi (X_val, Y_val) setelah pelatihan selesai. Berikut adalah penjelasan singkat dari setiap bagian kode:

1. final_loss, final_accuracy = model.evaluate(X_val, Y_val):
   - Menggunakan metode evaluate dari objek model untuk mengevaluasi performa model pada dataset validasi (X_val, Y_val).
   - X_val adalah input dari dataset validasi.
   - Y_val adalah target output (label) dari dataset validasi.
   - final_loss akan berisi nilai loss akhir dari model pada data validasi.
   - final_accuracy akan berisi akurasi akhir dari model pada data validasi.

2. print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy)):
   - Mencetak nilai akhir loss dan akurasi dari evaluasi model pada data validasi.

Ini adalah cara yang baik untuk mengukur kinerja model setelah pelatihan selesai. Pastikan model, X_val, dan Y_val telah didefinisikan dan siap digunakan sebelum menjalankan kode ini. Evaluasi seperti ini memberikan gambaran tentang seberapa baik model Anda dapat menggeneralisasi ke data yang tidak pernah dilihat sebelumnya (data validasi).



```
