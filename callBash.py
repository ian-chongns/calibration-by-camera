from os import system
from subprocess import check_output

dataFile = ('log.dat', 'coords.dat')

def appendFile(pattern, ind):
    try:
        if int(ind) == 0:
            system(f"LC_ALL=C echo $(date +%D\ %T) {pattern} >> {dataFile[int(ind)]}")
        else:
            system(f"LC_ALL=C echo {pattern} >> {dataFile[int(ind)]}")
    except:
        print("Error: Invalid Index")

def clear():
    system("LC_ALL=C clear")
