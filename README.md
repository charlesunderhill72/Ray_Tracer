# Mandelbrot Set Ray Tracer
A small Python program that computes a small user-specified region of the Mandelbrot Set and creates a 3D ray traced image of the region. 

## Features
- CLI with user-specified inputs
- Use of the time library to track computation and rendering time
- Error handling using python built-in error constructs

## Usage
### Setup
1. Clone this repo
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

### Execution
cd to the directory containing the ray_tracer.py script and run the program with:
```
python ./ray_tracer.py
```
Enter desired values for the text prompts. Refer to the project_paper.pdf file for some examples.

## TODO
- Rewrite code in C++ for more efficient computation
- Compute graphics using modern tools

## Citations
Peitgen, H.-O., Saupe, D., & Barnsley, M. F. (1988). The Science of Fractal Images. Springer-Verlag. 


