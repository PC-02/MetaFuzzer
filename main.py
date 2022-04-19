import multiprocessing
import subprocess
import os

from sympy import Mod

call = subprocess.check_output

def Fuzz():

    os.path.isdir("./size_seeds/") or os.makedirs("./size_seeds")  
    call(['./fuzzer','-i','size_in','-o','size_seeds','-l','10000','./size','@@'])


def Module():

    call(['python3', 'module.py', './size_seeds/', './size', 'NN', 'False', 'False'])
  

def Start():
    p1 = multiprocessing.Process(name='p1', target=Module)
    p = multiprocessing.Process(name='p', target=Fuzz)
    p1.start()
    p.start()

if __name__ == '__main__':
   Start()
