# -*- coding: utf-8 -*-
import config
import telebot

bot = telebot.TeleBot(config.token)

@bot.message_handler(content_types=["text"])
def repeat_all_messages(message): # Название функции не играет никакой роли, в принципе
    songName = message.text;
    #bot.send_message(message.chat.id, message.text)
    bot.send_message(message.chat.id, "test")

if __name__ == '__main__':
    bot.polling(none_stop=True)
