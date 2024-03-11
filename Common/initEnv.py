def spInitEnv():
    print("spInitEnv", __file__)
    # Access to env varibales
    #import os
    # Set up to read API keys from .env
    #install dotenv
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(".\env")             # now you have access to os.environ["HUGGINGFACEHUB_API_TOKEN"]

def getPath():
    return "Common" +  __file__
