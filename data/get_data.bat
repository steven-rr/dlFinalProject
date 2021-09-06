if exist data.zip del data.zip
if exist readme.txt del readme.txt
if exist Charts.zip del Charts.zip
if exist Charts_Mini.zip del Charts_Mini.zip
if exist compressedChartsEst.zip del compressedChartsEst.zip
if exist compressedChartsClose.zip del compressedChartsClose.zip
if exist compressedChartsVol.zip del compressedChartsVol.zip
if exist compressedChartsVol.zip del compressedChartsGRU.zip
if exist Charts\ rd /s /q charts
if exist Charts_Mini\ rd /s /q charts_mini
if exist compressedChartsEst\ rd /s /q compressedChartsEst
if exist compressedChartsClose\ rd /s /q compressedChartsClose
if exist compressedChartsVol\ rd /s /q compressedChartsVol
if exist compressedChartsGRU\ rd /s /q compressedChartsGRU

curl -L -o data.zip https://www.dropbox.com/sh/97451riinhod2t3/AACgi2DJdHgTfm2hdXfiguJ1a?raw=1
tar -zxvf data.zip -C .  
del data.zip

tar -zxvf Charts.zip -C .
del Charts.zip

tar -zxvf Charts_Mini.zip -C .
del Charts_Mini.zip

mkdir compressedChartsClose
tar -zxvf compressedChartsClose.zip -C .\compressedChartsClose
del compressedChartsClose.zip

mkdir compressedChartsEst
tar -zxvf compressedChartsEst.zip -C .\compressedChartsEst
del compressedChartsEst.zip

mkdir compressedChartsVol
tar -zxvf compressedChartsVol.zip -C .\compressedChartsVol
del compressedChartsVol.zip

mkdir compressedChartsGRU
tar -zxvf compressedChartsGRU.zip -C .\compressedChartsGRU
del compressedChartsGRU.zip

del readme.txt
