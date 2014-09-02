from os import getenv, path


def get_petsc_version():
    """Find the PETSc Git revision in petscconf.h"""
    dpath = '%s/include/petscconf.h' % getenv('PETSC_DIR')
    apath = '%s/%s/include/petscconf.h' % (getenv('PETSC_DIR'), getenv('PETSC_ARCH'))
    if path.exists(dpath):
        fname = dpath
    elif path.exists(apath):
        fname = apath
    else:
        return ""
    with open(fname) as f:
        for l in f:
            if 'define PETSC_VERSION_GIT' not in l:
                continue
            return l.split()[-1].strip('"')
