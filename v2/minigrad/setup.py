from setuptools import setup

setup(name='minigrad',
      version='0.1',
      description="Minimal auto differentiation engine",
      author='Justin Deschenaux',
      license='MIT',
      packages = ['minigrad'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=['numpy',],
      python_requires='>=3.8',
      )
