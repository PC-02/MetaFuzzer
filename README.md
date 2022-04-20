# MetaFuzzer

Fuzzer that allows for using multiple machine learning techniques during the fuzzing process.

## System Installation

Below are the dependencies

```
sudo apt-get update
sudo apt-get install -y build-essential python3-dev automake cmake git flex bison libglib2.0-dev libpixman-1-dev python3-setuptools
sudo apt-get install -y lld-11 llvm-11 llvm-11-dev clang-11 || sudo apt-get install -y lld llvm llvm-dev clang
sudo apt-get install -y gcc-$(gcc --version|head -n1|sed 's/\..*//'|sed 's/.* //')-plugin-dev libstdc++-$(gcc --version|head -n1|sed 's/\..*//'|sed 's/.* //')-dev
sudo apt-get install -y ninja-build 
sudo apt-get install -y tmux
```

Below are the python dependencies

```
python -m pip install --upgrade pip
python -m pip install -r requirements.py
```

During the fuzzing process,

```
CTRL + C # Stops the program
CTRL + D # Closes the split terminal
CTRL + B + [Arrow Key] # Move to terminal in direction of arrow key
```
