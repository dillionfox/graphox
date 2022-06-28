from setuptools import setup

setup(
    name='graphox',
    version='0.0.1',
    description='A simple implementation of Ollivier-Ricci curvature for NetworkX',
    url='https://github.com/dillionfox/graphox',
    author='Dillion Fox',
    author_email='11foxd1@gmail.com',
    license='GPLv2',
    packages=['graph_curvature'],
    install_requires=['pandas',
			'numpy',
			'networkit',
			'networkx',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GLPv3 License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
    ],
)
