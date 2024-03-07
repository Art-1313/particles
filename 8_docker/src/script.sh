#PREFIX_LIB = "/pythia8310/lib"
wget https://pythia.org/download/pythia83/pythia8310.tgz
tar -xvzf pythia8310.tgz
cd pythia8310
./configure
make
#export LD_LIBRARY_PATH=$(PREFIX_LIB):$LD_LIBRARY_PATH 