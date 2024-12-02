# Tracking dietary intake with image-based food recognition

## Description: 

## Setup

1. Create a conda environment and install the required dependencies:
```bash
mamba env create -f environment.yaml
```

2. Activate the environment:
```bash
mamba activate nutrition-tracker
```

3. Install bitsandbytes package from source to enable quantization:
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install .
```

4. Log in to Hugging Face from the terminal:
```
huggingface-cli login
```

### Participants (listed in alphabetical order by surname):
Pedro Andres Alba Diaz <br />
Troy Dutton <br />
Abinav Krishnan <br />
Haoji Liu <br />
Shashank Nag <br />

### Course:
ECE 3981K 3-Applied Machine Learning

### Semester:
Fall 2024

### Data:
Data is publically available from: 

### Website:
There is no website for this project. All information can be found in this repository.
