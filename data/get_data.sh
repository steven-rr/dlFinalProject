if [ -d ./Charts ]; then 
    rm -rf ./Charts
fi
if [ -d ./Charts_Mini ]; then 
    rm -rf ./Charts_Mini
fi
if [ -d ./compressedChartsEst ]; then 
    rm -rf ./compressedChartsEst
fi
if [ -d ./compressedChartsClose ]; then 
    rm -rf ./compressedChartsClose
fi
if [ -d ./compressedChartsVol ]; then 
    rm -rf ./compressedChartsVol
fi
if [ -d ./compressedChartsGRU ]; then 
    rm -rf ./compressedChartsGRU
fi
if [ -f ./data.zip ]; then
    rm ./data.zip
fi
if [ -f ./Charts.zip ]; then
    rm ./Charts.zip
fi
if [ -f ./Charts_Mini.zip ]; then
    rm ./Charts.zip
fi
if [ -f ./compressedChartsEst.zip ]; then
    rm ./compressedChartsEst.zip
fi
if [ -f ./compressedChartsClose.zip ]; then
    rm ./compressedChartsClose.zip
fi
if [ -f ./compressedChartsVol.zip ]; then
    rm ./compressedChartsVol.zip
fi
if [ -f ./compressedChartsGRU.zip ]; then
    rm ./compressedChartsGRU.zip
fi
if [ -f ./readme.txt ]; then
    rm ./readme.txt
fi
wget -O data.zip https://www.dropbox.com/sh/97451riinhod2t3/AACgi2DJdHgTfm2hdXfiguJ1a?raw=1
unzip data.zip
rm data.zip
unzip Charts.zip
rm Charts.zip
unzip Charts_Mini.zip
rm Charts_Mini.zip
mkdir compressedChartsEst
unzip compressedChartsEst.zip -d ./compressedChartsEst
rm compressedChartsEst.zip
mkdir compressedChartsClose
unzip compressedChartsClose.zip -d ./compressedChartsClose
rm compressedChartsClose.zip
mkdir compressedChartsVol
unzip compressedChartsVol.zip -d ./compressedChartsVol
rm compressedChartsVol.zip
mkdir compressedChartsGRU
unzip compressedChartsGRU.zip -d ./compressedChartsGRU
rm compressedChartsGRU.zip
rm readme.txt