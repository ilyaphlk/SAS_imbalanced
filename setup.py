from setuptools import setup, find_packages

setup(
    name="SAS_imbalanced",
    version="0.0.1",
    author="",
    author_email="",
    description="",
    long_description='',
    long_description_content_type="text/markdown",
    url='https://github.com/ilyaphlk/SAS_imbalanced',
    project_urls={
        "Bug Tracker": 'https://github.com/ilyaphlk/SAS_imbalanced/issues',
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #package_dir={"": "src"},
    #packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)