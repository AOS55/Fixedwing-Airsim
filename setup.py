import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FWAirsim",
    version="0.0.1",
    author="Alexander Quessy",
    author_email="aq15777@bristol.ac.uk",
    description="A python programme to accomplish various ML tasks with FW aircraft in AirSim",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AOS55/Fixedwing-Airsim",
    project_urls={
        "Bug Tracker": "https://github.com/AOS55/Fixedwing-Airsim/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8"
)
