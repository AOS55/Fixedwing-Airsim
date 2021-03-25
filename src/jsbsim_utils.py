import os
import sys
import shutil
import tempfile
import jsbsim


class Singleton:
    """
    A class used as a decorator to create a Singleton class

    ...

    Attributes
    ----------
    _cls : object
        the object instance
    _instance : object
        the instance of the class object

    Methods
    -------
    instance()
        Ensure the class can only be instantiated once
    """
    def __init__(self, cls):
        self._cls = cls

    def instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._cls()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through instance().')

    def __instancecheck__(self, instance):
        return isinstance(instance, self._cls)


class SandBox:
    """
    A class to create a temporary directory for Jsbsim's output

    ...

    Attributes
    ----------
    _tmpdir : str
        the path to the temporary directory to store output information
    path_to_jsbsim : str
        the absolute path to Jsbsim's root directory locally
    _relpath_to_jsbsim : str
        the relative path to Jsbsim's root directory from the temp folder

    Methods
    -------
    delete_csv_files()
        removes all the files contained within _tmpdir of type '.csv'
    path_to_jsbsim_file(*args)
        creates a link to the jsbsim root directory contained on the system
    exists(filename)
        searches to see if a filename exists within the temporary directory
    erase()
        erases the _tmpdir and its associated tree
    """
    def __init__(self, *args):
        self._tmpdir = tempfile.mkdtemp(dir=os.getcwd())
        path_to_jsbsim = os.path.join(os.path.dirname(sys.argv[0]), '/home/quessy/Dev/jsbsim', *args)
        self._relpath_to_jsbsim = os.path.relpath(path_to_jsbsim, self._tmpdir)

    def __call__(self, *args):
        return os.path.relpath(os.path.join(self._tmpdir, *args), os.getcwd())

    def delete_csv_files(self):
        files = os.listdir(self._tmpdir)
        for f in files:
            if f[-4:] == '.csv':
                os.remove(os.path.join(self._tmpdir, f))

    def path_to_jsbsim_file(self, *args):
        print('relpath is:', os.path.join(self._relpath_to_jsbsim, *args))
        return os.path.join(self._relpath_to_jsbsim, *args)

    def exists(self, filename):
        return os.path.exists(self(filename))

    def erase(self):
        shutil.rmtree(self._tmpdir)


class AttributeFormatter:
    ILLEGAL_CHARS = '\-/.'
    TRANSLATE_TO = '_' * len(ILLEGAL_CHARS)
    TRANSLATION_TABLE = str.maketrans(ILLEGAL_CHARS, TRANSLATE_TO)

    @staticmethod
    def translate(string: str):
        return string.translate(AttributeFormatter.TRANSLATION_TABLE)


def create_fdm(sandbox, pm=None):
    """
    A function that starts a jsbsim flight dynamic model and provides a path to each subsystem

    :param sandbox: the temp directory created by the Sandbox class
    :param pm: adds a relative path to the default sandbox jsbsim directory
    :return: the flight dynamic model
    """
    _fdm = jsbsim.FGFDMExec(os.path.join(sandbox(), ''), pm)
    print('FDM located at:', os.path.join(sandbox(), ''))
    path = sandbox.path_to_jsbsim_file()
    _fdm.set_aircraft_path(os.path.join(path, 'aircraft'))
    _fdm.set_engine_path(os.path.join(path, 'engine'))
    _fdm.set_systems_path(os.path.join(path, 'systems'))
    return _fdm
