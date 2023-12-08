"""Script for conversion from pgf figures to .pdf figures. Note, texlive needs to be installed."""
import argparse
import subprocess
import glob


def main(name):
    with open(name, "r") as f:
        x = f.readlines()

    if r"\end{document}" not in x:
        x.append(r"\end{document}")

    if r"\begin{document}" not in x:
        x.insert(0, r"\begin{document}")

    if r"\usepackage{pgfplots}" not in x:
        x.insert(0, r"\usepackage{pgfplots}")

    if r"\documentclass{standalone}" not in x:
        x.insert(0, r"\documentclass{standalone}")

        name = name.split(".pgf")[0]

    with open(name + ".tex", "w") as f:
        f.writelines(x)

    subprocess.Popen(["pdflatex", "%s.tex" % name]).communicate()[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="")

    args = parser.parse_args()
    if args.file == "":
        names = glob.glob("*.pgf")
        for name in names:
            main(name)

    else:
        main(args.file)
