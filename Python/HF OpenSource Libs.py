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
#                   model="./models/facebook/blenderbot-400M-distill")
chatbot = pipeline(task="conversational",
                   model="facebook/blenderbot-400M-distill")
