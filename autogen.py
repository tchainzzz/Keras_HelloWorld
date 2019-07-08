import TFInterface 
import inspect
import argparse
import contextlib
import sys
import subprocess

"""
    METHODS FOR INTROSPECTIVELY QUERYING DATA FROM MODULE

"""

def main_class_name():
    classes = inspect.getmembers(TFInterface, predicate=inspect.isclass)
    return [cls[0] for cls in classes if TFInterface.__name__ is cls[1].__module__][0]

def get_abstract_methods():
    classes = inspect.getmembers(TFInterface, predicate=inspect.isclass)
    return [cls[1] for cls in classes if TFInterface.__name__ is cls[1].__module__][0].__abstractmethods__

"""
    GENERIC CONSTANTS AND METHODS FOR REPEATED OPERATIONS

"""

GENERIC_SIGNATURE = "self"

def append_not_implemented_stub(f):
    f.write("        super().raise_override_error()\n\n")

def write_method(f, name):
    f.write("    def {}({}):\n".format(name, GENERIC_SIGNATURE))
    append_not_implemented_stub(f)

"""
    DIRECT FILE-WRITE METHODS

"""

def write_imports(f, classname):
    f.write("from {} import {}\n".format(TFInterface.__name__, classname))
    f.write("\n") 

def write_class_header(f, classname, superclassname):
    f.write("class {}({}):\n".format(classname, superclassname))
    f.write("    def __init__({}):\n".format(GENERIC_SIGNATURE))
    append_not_implemented_stub(f)

def write_class_methods(f, methods):
    for method in methods:
        write_method(f, method)

def write_main_stub(f, classname):
    f.write("if __name__ == '__main__':\n")
    f.write("    model = {}()".format(classname))

"""
    UTILITY
    Special thanks to Wolph on StackOverflow for providing this solution.

"""
@contextlib.contextmanager
def smart_open(filename, mode):
    if filename and filename != '-':
        f = open(filename, mode)
    else:
        f = sys.stdout
    try:
        yield f
    finally:
        if f is not sys.stdout:
            f.close()
    
if __name__ == '__main__':
    psr = argparse.ArgumentParser()
    psr.add_argument('-o', type=str, required=True, help="Output file name")
    psr.add_argument('--name', type=str, required=True, help="Name of the base class")
    psr.add_argument('--dry-run', action='store_true', default=False, help="Pipes output to stdout and does not save.")
    args = psr.parse_args()

    filename = args.o
    if not args.o.endswith(".py"):
        filename += ".py"
    if args.dry_run:
        filename = None
    with smart_open(filename, 'x') as f:
        write_imports(f, main_class_name())
        write_class_header(f, args.name, main_class_name())
        write_class_methods(f, get_abstract_methods())
        write_main_stub(f, args.name)
        funcs = inspect.getmembers(TFInterface.AbstractClassifier, predicate=inspect.isfunction)
        abs_funcs = [func[1] for func in funcs if hasattr(func[1], '__isabstractmethod__')]
    subprocess.run(["autopep8", "-i", filename])
    
    
