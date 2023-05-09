import subprocess


def get_num_lines(path):
    """counts number of lines in the given file

    Args:
        path (string): path to file

    Returns:
        int: number of lines
    """
    return int(subprocess.check_output(['wc', '-l', path]).split()[0])