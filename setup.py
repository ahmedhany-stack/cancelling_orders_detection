from setuptools import find_packages, setup

setup(
    name="src",
    version="1.0.0",
    author="Ahmed",
    author_email="ahmedhany126543@gmail.com",  # يمكنك تغييره لإيميلك الحقيقي
    description="An end-to-end Machine Learning pipeline for E-Retail Cancellation Prediction",
    long_description=open("README.md", encoding="utf-8").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/ahmedhany-stack/cancelling_orders_detection",  # رابط مستودع المشروع
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "scikit-learn",
    ],
)
