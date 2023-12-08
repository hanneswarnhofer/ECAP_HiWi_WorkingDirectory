import matplotlib as mpl
from os.path import join, dirname, abspath
import inspect
from tools.utils import to_list


def get_current_path():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    return dirname(abspath(filename))


def mplrc(rendering, formats_to_test=["png"]):
    assert len(formats_to_test) > 0, "At least one format has to be tested"
    path = get_current_path()
    mpl.rc_file(join(path, 'matplotlib_style_rc'))

    if rendering is False:
        print("do not render Latex fonts of plots\n fast plotting, but less nice")
    else:
        print("render Latex fonts of plots\n slower plotting")
        supported_formats = test_tex_install(formats=formats_to_test)

        if len(supported_formats) == 0:
            rendering = False
            mpl.rc('text', usetex=False)

    return rendering


def test_tex_install(log_dir="/tmp/", formats=["png", "pgf", "pdf", "svg"]):
    """
    Test texlive installation for various formats. Matplotlib requires specific tex packages for the rendering of fonts and the
    saving of the created figures.

    returns:
        formats which can be saved with tex formatting
    """
    print("testing tex install")
    import warnings
    formats = to_list(formats)
    import numpy as np
    from matplotlib import pyplot as plt
    working_formats = []

    x = np.random.randn(10)
    y = np.random.randn(10)

    fig, ax = plt.subplots(1)
    ax.scatter(x, y, label="$data$")

    x = np.random.randn(2)
    y = np.random.randn(2)
    ax.scatter(x, y, color="red", label="data red")
    ax.set_title(r"$\alpha$")
    ax.set_ylabel("$y / meters$")
    ax.set_xlabel("$x / meters$")
    ax.legend()

    def save(fig, w_list, format_):
        try:
            fig.savefig(join(log_dir + "/test.%s" % format_))
            w_list.append(format_)
        except (RuntimeError, FileNotFoundError, BrokenPipeError) as e:
            warnings.warn("Saving of %s format not possible because of missing install" % format_, UserWarning)
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
            pass

    for f in formats:
        save(fig, working_formats, f)

    plt.close(fig)
    del fig, ax

    return working_formats


def lower_(text):

    if text.count('_') == 1:
        if text.split('_')[1][0] != "{":
            text, lower = text.split('_')
            lower, _ = lower.split("$")
            text = text + "_{" + lower + "}" + "$"
    elif text.count('_') > 1:
        text = text.replace('_', r'\;')
    return text


def to_tex(txt):
    txt_l = txt.split("\n")

    text = ""
    for i, txt in enumerate(txt_l):
        if txt[-1] != '$':
            txt = txt + '}$'
        if txt[0] != '$':
            txt = r'$\mathrm{' + txt
        txt = lower_(txt)
        text = text + txt
    text = text.replace(' ', r'\;')

    return text.replace('$$', '$\n$')
