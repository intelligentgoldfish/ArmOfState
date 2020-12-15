# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:45:14 2020
@author: Thomas DeWitt
Contributor: Andrew Lu
"""

# ArmOfState2

import os
import discord
import csv
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
    
def save_scores():  # have it update every 10 minutes or something eventually
    with open(f"{dirs}/scores.csv", "w", newline="") as csvfile:
        save = csv.writer(csvfile)
        for pairs in toxiscores.items():
            save.writerow([str(pairs[0]), str(pairs[1])])

def load_preprocessing():
    with open(f"{dirs}/tokenizer2.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)  # load network tokenizer
    max_length = 200  # cut/pad all sentences to 200 tokens (words)
    trunc_type = 'post' #leave as-is
    padding_type = 'post' #leave as-is
    penalizing_threshold = 0.27 #min individual
    modifier = lambda x : np.round(pow(1.075, -3/8 * x), 2)
    gain = 1
    loss = 1 # Toxicity score modifier & factor modifier should be multiplied by for gaining/losing toxicity, 1 gain/loss default
    return tokenizer, max_length, trunc_type, padding_type, penalizing_threshold, modifier, gain, loss

# Load model and hyperparameters
model = tf.keras.models.load_model(f'{dirs}/classifier/AoS_GPnet_V2') #leave as-is
tokenizer, max_length, trunc_type, padding_type, penalizing_threshold, modifier, gain, loss = load_preprocessing()

# scoring and score management
def score_text(toxic_preds):
    if max(toxic_preds) <= penalizing_threshold:
        score = 0
    else:
        scale = max(toxic_preds)
        rank = np.where(toxic_preds == scale)
        rank = rank[0]
        rank = rank[0] + 1
        score = np.abs(scale) * rank
    return score

def manage_toxiscores(uid, message_score):
    if uid not in toxiscores.keys():
        toxiscores[uid] = 0.00
    
    user_score = toxiscores[uid]
    message_score = np.abs(message_score)
    
    if message_score <= 0.00:  # Losing toxicity
        user_score -= modifier(user_score) * loss
        user_score = 0.00 if toxiscores[uid] <= 0.00 else user_score
    else:  # Gaining toxicity
        user_score += np.abs(modifier(user_score) * message_score * gain)

    toxiscores[uid] = np.round(user_score, 2)
    return toxiscores


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
            #stringlist = []
            #model.summary(print_fn=lambda x: stringlist.append(x))
            #short_model_summary = "\n".join(stringlist)
            #await message.channel.send(short_model_summary)
            
    
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
        words = str(message.content)
        toxic_tokens = tokenizer.texts_to_sequences(words)
        toxic_ready = pad_sequences(toxic_tokens, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        toxic_preds = model.predict(toxic_ready)
        toxic_results = np.amax(toxic_preds, axis=0)
        toxic_score = score_text(toxic_results)
        if debug_mode:
            print("Raw model out: ",str(toxic_preds))
            print(str(toxic_results))
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
