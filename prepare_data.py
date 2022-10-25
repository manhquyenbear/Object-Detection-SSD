from genericpath import exists
import os
import urllib.request
import zipfile

data_dir="./data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

url= "https://data.deepai.org/PascalVOC2012.zip"

target_path = os.path.join(data_dir, "VOC.zip")
if not os.path.exists(target_path) : # xem có tồn tại file chưa
    urllib.request.urlretrieve(url, target_path) #download 

    zip = zipfile.ZipFile(target_path) #đọc file zip
    zip.extractall(data_dir) # giải nén vào datadir
    zip.close
