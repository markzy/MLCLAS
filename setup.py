try:
    from setuptools import setup, find_packages
except ImportError:
    import ez_setup
    ez_setup.use_setuptools()
    from setuptools import setup, find_packages

setup(
    name='MLCLAS',
    version='0.0.3',
    description='Multi-label classification algorithms implemented in Python',
    author='Mark Zhou',
    author_email='zhouyao0808@gmail.com',
    packages=find_packages(exclude=["examples", "*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=['numpy>=1.10', 'scipy>=0.16.0', 'scikit-learn>=0.17', 'cvxpy>=0.4.0'],
    license="MIT",
    url="https://github.com/markzy/multi-label-classification",
    keywords="multi label classification",
)
