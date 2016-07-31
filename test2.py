# -*- coding: utf-8 -*-
import config
import telebot
import os

# state = 0;

bot = telebot.TeleBot(config.token)
songName = None

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Введи название песни и я поищу её в своей базе")
    config.state = 0


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    global songName
    if (config.state == 0):
        songName = message.text
        bot.reply_to(message, "Теперь введи название стиля")
        config.state += 1
    elif (config.state == 1):
        stalyName = message.text
        config.state += 1

        os.system("python ~/Documents/Hackaton/coffee3p1_chatbot/Run.py --subject %s --style %s" % (songName, stalyName))
        audio = open('result.wav', 'rb')
        bot.send_audio(message.chat.id, audio)


bot.polling()
