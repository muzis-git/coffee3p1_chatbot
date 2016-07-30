# -*- coding: utf-8 -*-
import config
import telebot

#state = 0;

bot = telebot.TeleBot(config.token)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Введи название песни и я поищу её в своей базе")
    config.state = 0;

@bot.message_handler(func=lambda message: True)
def echo_all(message):    
    if(config.state == 0):
        songName = message.text;        
        bot.reply_to(message, "Теперь введи название стиля");
        config.state += 1;
    if(config.state == 1):
        stalyName = message.text;
        config.state += 1;

bot.polling()
