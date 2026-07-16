from setuptools import setup, find_packages

setup(
    name="kernelgenbench",
    version="0.1.0",
    description="Benchmark framework for Triton kernel generation and verification",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    include_package_data=True,
    install_requires=[
        "anthropic>=0.71.0",
        "openai>=2.24.0",
        "vllm==0.13.0",
        "fastapi",
        "uvicorn",
        "PyYAML",
        "scipy",
        "rich",
        "tqdm",
        "ijson",
    ],
)
