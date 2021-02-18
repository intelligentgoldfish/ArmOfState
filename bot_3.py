"""
Created on Wed Feb 11 10:09:43 2021
Author: Thomas DeWitt
Contributor: Andrew Lu

Version 1.2 Alpha
New features: 
-TFLite models to accelerate inference and reduce memory consumption

Any multi-line comments are reserved for future editions or version transitions of the bot, 
and may not be changed, implemented, or removed without the author's express permission.
"""

import os
import discord
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings

print(tf.__version__)
print(keras.__version__)

print('All modules successfully loaded.  Initializing...')

# Path
dirs = os.getcwd()

# Discord token
def read_token():
    with open(f"{dirs}/token.txt", "r") as file:
        lines = file.readlines()
        return lines[0].strip()

# Format: <string uid> <float 2 score>\n
def load_scores():
    scores = {}
    try:
        with open(f"{dirs}/scores.csv", "r", newline="") as csvfile:
            lines = csv.reader(csvfile)
            for line in lines:
                scores[str(line[0])] = float(line[1])
        return scores
    except:
        with open(f"{dirs}/scores.csv", "w"):
            return scores

# handle score saving when shutting down
def save_scores():  # have it update every 10 minutes or something eventually
    with open(f"{dirs}/scores.csv", "w", newline="") as csvfile:
        save = csv.writer(csvfile)
        for pairs in toxiscores.items():
            save.writerow([str(pairs[0]), str(pairs[1])])

# load classifier parameters
def load_preprocessing():
    binary_threshold = 0.55 # binary model toxicity threshold
    multi_threshold = 0.55 # multi-class penalty threshold
    modifier = lambda x : np.round(pow(1.075, -3/8 * x), 2)
    gain = 1
    loss = 1
    return binary_threshold, multi_threshold, modifier, gain, loss

# score text with classifier predictions
def score_text(toxic_pred, msg):
    toxic_pred = toxic_pred[0]
    if toxic_pred <= binary_threshold:
        score = 0
    else:
        score = toxic_pred
        penalties = conduct_inference(V5_multi, multi_input_details, multi_output_details, msg)
        for item in penalties:
            if item >= multi_threshold:
                score += item
    return score

# update scores for each user
def manage_toxiscores(uid, message_score):
    if uid not in toxiscores.keys():
        toxiscores[uid] = 0.00
    
    user_score = toxiscores[uid]
    if message_score < 0:
        warnings.warn(f"Message score cannot be less than 0.  Received score of {message_score}")
        message_score = np.abs(message_score)
    
    user_score += np.abs(modifier(user_score) * message_score * gain)

    toxiscores[uid] = np.round(user_score, 2)
    return toxiscores

# the interpreter basically functions as a tokenizer, but simpler
# separate model because tflite doesn't support lookup tables
def load_interpreter(interpreter_path):
    interpreter_model = tf.keras.models.load_model(interpreter_path)
    return interpreter_model

# initialize high-speed TFLite model
def prepare_model_for_inference(path):
    inference_model = tf.lite.Interpreter(model_path=f"{path}.tflite")
    inference_model.allocate_tensors()
    input_details = inference_model.get_input_details()
    output_details = inference_model.get_output_details()
    input_shape = input_details[0]['shape']
    return inference_model, input_details, output_details, input_shape

# conduct inference with prepared TFLite model
def conduct_inference(inference_model, input_details, output_details, data):
    inference_model.set_tensor(input_details[0]['index'], data)
    inference_model.invoke()
    outputs = inference_model.get_tensor(output_details[0]['index'])
    return outputs


binary_path = f'{dirs}/classifier/AoS_V5_binary'
multi_path = f'{dirs}/classifier/AoS_V5_multi'
interpreter_path = f'{dirs}/classifier/AoS_V5_interpreter'

binary_threshold, multi_threshold, modifier, gain, loss = load_preprocessing()
V5_binary, binary_input_details, binary_output_details, binary_input_shape = prepare_model_for_inference(binary_path)
V5_multi, multi_input_details, multi_output_details, multi_input_shape = prepare_model_for_inference(multi_path)
V5_interpreter = load_interpreter(interpreter_path)


# Tokens
token = read_token()

# Master users
master_users = ["641816848865689611"]

# List of commands
commands = ["command (authorization): function",
            "^help (standard): Call command list",
            "^hello (standard): Greet the bot",
            "^watchlist (standard): Get top 20 on toxicity watchlist",
            "!#init (master): Load model and preprocessing methods",
            "!#startup (master): Run model",
            "!#standby (master): Pause analysis while keeping bot online",
            "!#scrape (master): Initialize data scraping (for later use in unsupported features)",
            "!#shutdown (master): Terminate runtime on EC2 instance",
            "!#debug (master): Initialize debug mode",
            "!#nodebug (master): Deactivate debug mode"
           ]

# Scraping info (Eventually write to file instead)
messages = []
authors = []

# Model statuses
scrape_messages = False
prep_to_analyze = False
model_ready = False
debug_mode = False

# Load toxicity scores
toxiscores = load_scores()

client = discord.Client()

print('Connected to client.')

@client.event
async def on_message(message):
    # we need these lmao
    global scrape_messages
    global prep_to_analyze
    global model_ready
    global toxiscores
    global debug_mode

    # Don't talk to bots (including self)
    if message.author.bot:
        return
    
    if message.content.startswith("!#debug"):
        debug_mode = True
    
    if message.content.startswith("!#nodebug"):
        debug_mode = False
    
    if message.content.startswith("!#scrape"):
        if str(message.author.id) in master_users:
            scrape_messages = True
            await message.channel.send("WARNING: all messages following this message will be filed for training purpose.")
    
    if message.content.startswith("!#init"):
        if str(message.author.id) in master_users:
            model_ready = True
            await message.channel.send("Model initialized.  Standing by.")
            
    
    if message.content.startswith("!#startup"):
            if str(message.author.id) in master_users:
                if model_ready == True:
                    prep_to_analyze = True
                    await message.channel.send("Analysis framework active.")
                else:
                    await message.channel.send("Model not loaded. Please load model.")

    if message.content.startswith("!#shutdown"): #shutdown bot and disconnect from all servers
        if str(message.author.id) in master_users:
            await message.channel.send("Saving data...")
            data = pd.DataFrame(data={"user": authors, "message": messages})
            data.to_csv(f"{dirs}/discord_messages.csv", sep=",", index=False)  # Change this eventually
            save_scores()
            await message.channel.send("Tschuss.")
            await client.logout()
            
    if message.content.startswith("!#standby"): #prevent model from running without terminating bot (for example, if burning too much cloud memory)
        if str(message.author.id) in master_users:
            await message.channel.send("Deactivating analysis framework.")
            prep_to_analyze = False
    
    if scrape_messages == True:  # Eventually directly write to a buffer then file, also maybe only from one channel (specific channel id)
        messages.append(str(message.content))
        authors.append(str(message.author.id))
        
    if prep_to_analyze == True:
        words = tf.constant([words]) # set words as tensors
        words = V5_interpreter.predict(words) # vectorize for tflite embeddings
        toxic_preds = conduct_inference(V5_binary, binary_input_details, binary_output_details, words) # run tflite inference
        toxic_score = score_text(toxic_preds, words) # score text
        if debug_mode:
            print("Raw model out: ",str(toxic_preds))
            print("Scored results: ",str(toxic_score))
        
        if debug_mode:
            print("Raw model out: ",str(toxic_preds))
            print(str(toxic_preds))
            print(str(toxic_score))
        offender = str(message.author.id)
        toxiscores = manage_toxiscores(offender, toxic_score)

    if message.content == "^hello":
        reply = "Greetings, citizen."
        await message.channel.send(reply)
        
    if message.content == "^help":
        reply = ""
        for command in commands:
            reply += f"{command} \n"
        await message.channel.send(reply)

    if message.content == "^watchlist":  # Untested
        tmp = dict(sorted(toxiscores.items(), key=lambda x: x[1], reverse=True))
        tmp_uid = [uid for uid, score in tmp.items()]
        tmp_scores = [score for uid, score in tmp.items()]

        reply = "{: <40}".format("```Toxicity Watchlist") + ": Score\n"
        
        place = 0
        while place < 20 and place < len(tmp):
            reply += "{: <40}".format(f"{str(place+1)}. {str(client.get_user(int(tmp_uid[place])))}") + ": {:2.2f}\n".format(tmp_scores[place])
            place += 1
        await message.channel.send(reply + "```")
    
    

client.run(token)

print('Connection closed.')