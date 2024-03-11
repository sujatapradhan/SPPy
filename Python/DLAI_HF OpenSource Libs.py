#Ref https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/3/natural-language-processing-(nlp)
# as sp@gmail
'''
Lesson 2: Natural Language Processing (NLP)
In the classroom, the libraries are already installed for you.
If you would like to run this code on your own machine, you can install the following:
 !pip install transformers
Build the chatbot pipeline using 🤗 Transformers Library
'''

# Here is some code that suppresses warning messages.
# !pip install transformers
### from transformers.utils import logging
### logging.set_verbosity_error()

from transformers import pipeline
#import tensorflow 
#chatbot = pipeline(task="conversational",
#                   model="./models/facebook/blenderbot-400M-distill") #FOR LOCAL on LAPTOP
chatbot = pipeline(task="conversational",
                   model="facebook/blenderbot-400M-distill") #had to pip install torch and then tf-keras - lot of warnings
#-------------------------------------------------------------
user_message = """
What are some fun activities I can do in the winter?
"""
from transformers import Conversation
conversation = Conversation(user_message)
print(conversation)
#-------------------------------------------------------------
conversation = chatbot(conversation)
print(conversation)
#-------------------------------------------------------------
conversation.add_message(
    {"role": "user",
     "content": """
What else do you recommend?
"""
    })
print(conversation)
#-------------------------------------------------------------
conversation = chatbot(conversation)
print(conversation)
########################################################################
# Free up some memory before continuing
# In order to have enough free memory to run the rest of the code, please run the following to free up memory on the machine.
import gc
del conversation
gc.collect()




####################################################################
from transformers import pipeline 
import torch
translator = pipeline(task="translation", # translation, "zero-shot-audio-classification",conversational, summarization
                      model="facebook/nllb-200-distilled-600M", # no loang left behind
                      torch_dtype=torch.bfloat16)  #compress model w/o perf degradation
# To choose other languages, you can find the other language codes on the page: Languages in FLORES-200 
# https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

#----------------------------------------------------------------

#NLLB: No Language Left Behind: 'nllb-200-distilled-600M'.

text = """\
My puppy is adorable, \
Your kitten is cute.
Her panda is friendly.
His llama is thoughtful. \
We all have nice pets!"""
text_translated = translator(text,
                             src_lang="eng_Latn",
                             #tgt_lang="fra_Latn"
                             tgt_lang="hin_Deva"
                             )
text_translated


########################################################################
# Free up some memory before continuing
import gc
del translator
gc.collect()

########################################################################
# Build the summarization pipeline using 🤗 Transformers Library
summarizer = pipeline(task="summarization",
                      model="facebook/bart-large-cnn",
                      torch_dtype=torch.bfloat16)
# ---------------------------------------------------------------------
 
text = """Paris is the capital and most populous city of France, with
          an estimated population of 2,175,601 residents as of 2018,
          in an area of more than 105 square kilometres (41 square
          miles). The City of Paris is the centre and seat of
          government of the region and province of Île-de-France, or
          Paris Region, which has an estimated population of
          12,174,880, or about 18 percent of the population of France
          as of 2017."""
summary = summarizer(text,
                     min_length=10,
                     max_length=100)
summary

#---------------------------------------------------------------
# Free up some memory before continuing
import gc
del summarizer
gc.collect()





########################################################################


# Build the sentence embedding pipeline using 🤗 Transformers Library
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
# -----------------------------------------------------------

sentences1 = ['The cat sits outside',
              'A man is playing guitar',
              'The movies are awesome']
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings1
sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']
embeddings2 = model.encode(sentences2, 
                           convert_to_tensor=True)
print(embeddings2)
#-----------------------------------------------------------------
#Calculate the cosine similarity between two sentences as a measure of how similar they are to each other.
from sentence_transformers import util
cosine_scores = util.cos_sim(embeddings1,embeddings2)
print(cosine_scores)
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i],
                                                 sentences2[i],
                                                 cosine_scores[i][i]))
#---------------------------------------------------------------
# Free up some memory before continuing
import gc
del model
gc.collect()



######################################################################
#Lesson 5: Zero-Shot Audio Classification
'''
  !pip install transformers
  !pip install datasets
  !pip install soundfile
  !pip install librosa
The librosa library may need to have ffmpeg installed.
'''

#Here is some code that suppresses warning messages.
#from transformers.utils import logging
#logging.set_verbosity_error()

#--------------------------------------------------
# Prepare the dataset of audio recordings
from datasets import load_dataset, load_from_disk
# This dataset is a collection of different sounds of 5 seconds
dataset = load_dataset("ashraq/esc50",
                       split="train[0:10]")
# dataset = load_from_disk("./models/ashraq/esc50/train")
#------------------------------------------------------------
#### USE len(), type() 
audio_sample = dataset[0]
audio_sample
from IPython.display import Audio as IPythonAudio
IPythonAudio(audio_sample["audio"]["array"],
             rate=audio_sample["audio"]["sampling_rate"])
#---------------------------------------------------------------
#Build the audio classification pipeline using 🤗 Transformers Library
from transformers import pipeline
zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="laion/clap-htsat-unfused")

#--------------------------------------------------------------------
# Sampling Rate for Transformer Models
# 12=How long does 1 second of high resolution audio (192,000 Hz) appear 
# to the Whisper model (which is trained to expect audio files at 16,000 Hz)?
# The 1 second of high resolution audio a=  12 seconds of audio to whisper.
(1 * 192000) / 16000
# 60=How about 5 seconds of audio?
(5 * 192000) / 16000

zero_shot_classifier.feature_extractor.sampling_rate
audio_sample["audio"]["sampling_rate"]
# Cast and Set the correct sampling rate for the input and the model.
from datasets import Audio
dataset = dataset.cast_column(
    "audio",
     Audio(sampling_rate=48_000))
audio_sample = dataset[0]
audio_sample
candidate_labels = ["Sound of a dog",
                    "Sound of vacuum cleaner"]
zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels)
candidate_labels = ["Sound of a child crying",
                    "Sound of vacuum cleaner",
                    "Sound of a bird singing",
                    "Sound of an airplane"]
zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels)
# This appraoch tries to find teh most plausibel match

#---------------------------------------------------------------
# Free up some memory before continuing
import gc
del zero_shot_classifier
gc.collect()

