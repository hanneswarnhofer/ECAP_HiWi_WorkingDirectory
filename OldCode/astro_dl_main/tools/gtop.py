#!/usr/bin/python2.7
import time
from subprocess import Popen, PIPE


UP = "\x1b[1A"
ERASE = "\x1b[2K"


def gtop(erase):
    p = Popen("nvidia-smi", stdout=PIPE, stderr=PIPE, shell=True, executable="/bin/bash")
    out, err = p.communicate()
    if p.returncode != 0:
        raise Exception(err)

    print((UP + ERASE) * erase + UP)

    print(out)
    return len(out.split("\n"))


def main():
    try:
        erase = 0
        while True:
            erase = gtop(erase)
            time.sleep(2)
    except KeyboardInterrupt:
        print("")


if __name__ == "__main__":
    main()
