cd daknet
sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
sed -i 's/GPU=0/GPU=1/g' Makefile
make
./darknet

cd cfg
wget https://pjreddie.com/media/files/darknet53.conv.74
wget https://pjreddie.com/media/files/yolov3.weights