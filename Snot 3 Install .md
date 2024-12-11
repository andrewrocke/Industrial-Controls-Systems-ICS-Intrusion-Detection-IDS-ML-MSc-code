install
https://shape.host/resources/how-to-install-and-configure-snort-3-intrusion-detection-system-on-ubuntu-22-04

https://blog.snort.org/2024/03/talos-launching-new-machine-learning.html

Install guide from here clean ubutnu 22 VM.
Install dependencies

cmake to build from sourcegit verion
sudo apt install cmake
sudo apt install git
sudo apt install g++
sudo apt-get install autoconf
sudo apt install libtool
didnt work

daq from https://github.com/snort3/libdaq for packet IO

sudo git clone https://github.com/snort3/libdaq.git
cd libdaq
./bootstrap./



tried this sudo apt install -y gcc libpcre3-dev zlib1g-dev libluajit-5.1-dev 
libpcap-dev openssl libssl-dev libnghttp2-dev libdumbnet-dev 
bison flex libdnet autoconf libtool
still didnt work 
tried this
sudo apt install pkg-config

this worked after this pkg-config 
./bootstrap./

run as 3 separate command 
sudo ./configure
sudo make
sudo make install

check with https://github.com/ofalk/libdnet

next https://github.com/ofalk/libdnet

run

sudo git clone https://github.com/ofalk/libdnet

run 

cd libdnet

found instructions in the INSTALL directory 


\(ran ./configure \&\& sudo make in Libdnet\)
this failed om check not being available 

so ran sudo apt install check

ran ./configure \&\& sudo make in Libdnet

from: https://github.com/snort3/snort3
git clone https://github.com/snort3/snort3.git
cd snort3/
$export my_path=/path/to/snorty$

$./configure_cmake.sh --prefix=$ $my_path$

needs CMAKe 
sudo apt install cmake

rerun
$sudo ./configure_cmake.sh --prefix=$$my_path$

got error

ran 
sudo apt-get install libdumbnet-dev

ran again 
$sudo ./configure_cmake.sh --prefix=$$my_path$

got error
ran 
sudo apt-get install -y libhwloc-dev

reran 
$sudo ./configure_cmake.sh --prefix=$ $my_path$

got error SSL
ran 
sudo apt-get install libssl-dev

got error pcap
ran 
 sudo apt-get install libpcap-dev

reran 
$sudo ./configure_cmake.sh --prefix=$$my_path$

error tired 
sudo apt install libpcre3 libpcre3-dev
but it was uptodate so tried
sudo apt-get install luajit pkg-config - whcih installed

still cannot build
installed 
sudo apt-get install libhyperscan-dev
and
sudo apt-get install libsafe-iop0 
sudo apt-get install uuid-dev
sudo apt-get install libunwind-dev

 ran agin built
 
$sudo ./configure_cmake.sh --prefix=$$my_path$

then cd build
$then sudo make -j $$(nproc) install$


dnet from https://github.com/dugsong/libdnet.git for network utility functions
flex >= 2.6.0 from https://github.com/westes/flex for JavaScript syntax parser
g++ >= 5 or other C++14 compiler
hwloc from https://www.open-mpi.org/projects/hwloc/ for CPU affinity management
LuaJIT from http://luajit.org for configuration and scripting
$OpenSSL from https://www.openssl.org/source/ for SHA and MD5 file signatures, the protected_content rule option, and SSL service detection$
pcap from http://www.tcpdump.org for tcpdump style logging
pcre from http://www.pcre.org for regular expression pattern matching
pkgconfig from https://www.freedesktop.org/wiki/Software/pkg-config/ to locate build dependencies
zlib from http://www.zlib.net for decompression
