import sys
sys.path.append('C:/Users/a138821/OneDrive - Eviden/Sujata/Dynamic/Work/SPCode/SPPy/')
# load a source module from a file
import Common
Common.initEnv.spInitEnv()


#########################
#https://www.youtube.com/watch?v=TD9JzIiOlBo&t=128s

# Quantized Llama 2 models from HF from TheBloke with ctransformer package directly

#!pip install ctransformers>=0.2.24
# https://huggingface.co/TheBloke/Llama-2-13B-Ensemble-v6-GGUF

from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-Ensemble-v5-GGUF", model_file="llama-2-13b-ensemble-v5.Q5_K_M.gguf", model_type="llama", gpu_layers=0)

