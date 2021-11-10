#!/usr/bin/env python
# coding: utf-8

# # 8.  Bot Telegram

# In[1]:


import pandas as pd


# In[2]:


import os


# In[3]:


os.chdir("D:/AI/NLP")


# In[4]:


from NLP_Models import model_prediction as dhsd
import logging
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler


# In[5]:


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


# In[6]:


logger = logging.getLogger(__name__)


# In[7]:


def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi guys, Yuk deteksi ucapan yang kamu gunakan sehari hari disini!')


# In[8]:


def nlpResult(update: Update, context: CallbackContext):
    hate = dhsd.hateSpeechPredict(update.message.text)
    result = hate['final_result']
    confidence = hate['confidence']
    update.message.reply_text('Wuah ucapan' + '\t' + str(update.message.text) + ' termasuk dalam kategori :' +'\t' +str(result) +'\t' +'dengan confidence ' +str(confidence))


# In[ ]:


def main():
    updater = Updater('1882556776:AAHdjrBDbO7RJYrJMv9EUYQ99DmF59OnTXM')
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.text, nlpResult))
    updater.start_polling()
    updater.idle()
if __name__ == '__main__':
    main()


# In[ ]:




