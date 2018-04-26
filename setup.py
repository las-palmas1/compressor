from distutils.core import setup

setup(
    name='compressor',
    version='0.0.1',
    package_dir={'compressor': 'core', 'compressor.average_streamline': 'core/average_streamline',
                 'compressor.templates': 'core/templates'},
    packages=['compressor', 'compressor.average_streamline', 'compressor.templates'],
    package_data={'compressor': ['templates/average_streamline.tex']},
    url='',
    license='',
    author='Alexander Zhigalkin',
    author_email='',
    description=''
)
