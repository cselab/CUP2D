git clone https://github.com/oneapi-src/oneTBB.git
cd oneTBB
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../../my_installed_onetbb -DTBB_TEST=OFF ..
cmake --build . --config relwithdebinfo
cmake --install . --config relwithdebinfo
