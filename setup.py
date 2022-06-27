from setuptools import setup

setup(
    name='graph_curvature',
    version='0.0.1',    
    description='A simple implementation of Ollivier-Ricci curvature for NetworkX',
    url='https://github.com/dillionfox/graph_curvature',
    author='Dillion Fox',
    author_email='11foxd1@gmail.com',
    license='GPLv2',
    packages=['graph_curvature'],
    install_requires=['mpi4py>=2.0',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
