from setuptools import setup

setup(name='deep_learn',
      version='0.1.0',
      description='Deep Learning Package',
      long_description=open('README.md').read(),
      url='https://github.com/itratrahman/deep_learn',
      author='Itrat Rahman',
      author_email='rahmanitrat@gmail.com',
      license='BSD 3-Clause License',
      packages=['deep_learn', 'deep_learn.nn', 'deep_learn.utils'],
      install_requires=["numpy"],
      zip_safe=False)
