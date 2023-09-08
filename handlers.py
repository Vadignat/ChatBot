import torch

from main import bot, dp
from aiogram.types import Message
from config import admin_id
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '.\saved_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_response(input_text, model, tokenizer, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

async def send_to_admin(dp):
    await bot.send_message(chat_id=admin_id, text="Бот запущен")


@dp.message_handler(commands=['start'])
async def process_start_command(message: Message):
    await bot.send_message(chat_id=message.from_user.id,
                           text=f"Привет, {message.from_user.first_name}!")
    await bot.send_message(chat_id=admin_id,
                           text=f"{message.from_user.username} запустил бота: https://t.me/{message.from_user.username}")


@dp.message_handler()
async def echo(message: Message):
    text = message.text
    generated_response = generate_response(text, model, tokenizer, max_length=len(text) + 20)
    await message.answer(text=generated_response)






