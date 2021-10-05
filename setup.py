from urllib.request import urlretrieve
import zipfile

download_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip'
donwload_path = 'vec.zip'

print('downloading fasttext')
urlretrieve(download_url, donwload_path)

print('unzipping...')
with zipfile.ZipFile(donwload_path, 'r') as zip_ref:
    zip_ref.extractall()

print('done')
