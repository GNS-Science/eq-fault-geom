from setuptools import setup, find_packages

setup(name='eq-fault-geom',
      version='0.1',
      description='Python tools for manipulation of fault sources for the NSHM',
      author='NHSM project team',
      author_email='a.howell@gns.cri.nz',  # Very happy to change this
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      )
