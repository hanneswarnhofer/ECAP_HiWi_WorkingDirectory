"""Script for exporting conda env including version numbers and pip installations."""
import argparse
import subprocess
import yaml


def main(args):
    full_env = yaml.safe_load(subprocess.check_output("conda env export --no-builds".split(" ")))

    if 'conda-forge' not in full_env["channels"]:
        full_env["channels"] += ["conda-forge"]

    history_env = yaml.safe_load(subprocess.check_output("conda env export --from-history".split(" ")))
    history_env["name"] = args.name
    history_env["channels"] = full_env["channels"]

    try:
        history_env.pop("prefix")
    except KeyError:
        pass

    try:
        history_env.pop("main")
    except KeyError:
        pass

    try:
        history_env.pop("variables")
    except KeyError:
        pass

    packages = history_env["dependencies"]
    # packages = [p for p in packages if "python==" not in p]  # no python install

    if args.version is True:
        p_new = []

        for k in packages:

            if "=" in k:
                p_new.append(k)
                continue
            elif k == "pytorch":
                exec("import torch")
                version = eval("torch.__version__")
                p_new.append("%s=%s" % (k, version))
                continue

            try:
                exec("import %s" % k)
                try:
                    version = eval("%s.__version__" % k)
                except AttributeError:
                    version = eval("%s.version" % k)

                p_new.append("%s=%s" % (k, version))
            except:  # noqa
                p_new.append(k)
                pass

        packages = p_new

    if "pip" not in packages:
        import pip
        packages.append("pip=%s" % pip.__version__)

    if args.cuda is True:
        packages.append({"pip": ["-r requirements_pip.txt"]})
    else:
        packages.append({"pip": ["-r requirements_pip_nocuda.txt"]})

    history_env["dependencies"] = packages

    with open(args.file_name, "w") as f:
        yaml.safe_dump(history_env, f, sort_keys=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="astro_dl")
    parser.add_argument("--file-name", type=str, default="./env.yml")
    parser.add_argument("--version", type=bool, default=True)
    parser.add_argument("--cuda", type=bool, default=True)

    args = parser.parse_args()
    print("exporting CONDA env")
    main(args)
