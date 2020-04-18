import logging

import telebot

from settings import token
from utils import download_image, predict

EXAMPLE='https://live.staticflickr.com/3222/2899478544_929450365d_b.jpg'

bot = telebot.TeleBot(token)
logger = telebot.logger
telebot.logger.setLevel(logging.INFO)


@bot.message_handler(commands=['start', 'help'])
def start(message):
	bot.reply_to(
        message,
        f'👋Hi!\nSend me picture with house number as *photo*, like [this]({EXAMPLE}):',
        parse_mode='Markdown'
    )


@bot.message_handler(content_types=['photo'])
def photos_handler(message):
    photo_info = message.photo[0]
    try:
        file = bot.get_file(photo_info.file_id)
        image = download_image(file.file_path)
        number = predict(image)
        bot.reply_to(message, f'I think this is {number}')
    except:
        bot.reply_to(message, 'Sorry, I can\t recognize this')

bot.infinity_polling()
