!pip install -U transformers bitsandbytes>=0.39.0 -q

!pip install pyTelegramBotAPI

!pip install -U deep-translator -q

from deep_translator import GoogleTranslator

!pip install accelerate
!pip install -i https://pypi.org/simple/ bitsandbytes

!pip install g4f -q

!pip install diffusers -q

import g4f
from g4f.Provider import Bing, OpenaiChat, Liaobots, BaseProvider
from g4f.cookies import set_cookies
from g4f.client import Client
from deep_translator import GoogleTranslator
import nest_asyncio

nest_asyncio.apply()

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import set_seed
import torch
name = 'facebook/opt-350m'
name1 = "mistralai/Mistral-7B-Instruct-v0.1"
name2 = "HuggingFaceH4/zephyr-7b-alpha"
name3 = 'openchat/openchat_3.5'
#, device_map="auto", load_in_4bit=True)
tokenizer_ = AutoTokenizer.from_pretrained(name2)
model = AutoModelForCausalLM.from_pretrained(name2, device_map='auto', load_in_4bit=True)
#'эту строку использовать только для name2 - load_in_4bit=True

conversation_history = [{"role": "user", "content": ""},
{"role": "assistant", "content": ""}]

def getResponse(text):
    set_seed(0)
    conversation_history.append({'role': 'user', 'content': text})
    
    model_inputs = tokenizer_.apply_chat_template(conversation_history[-40:], add_generation_prompt=True, return_tensors="pt").to("cuda")
    input_length = model_inputs.shape[1]
    #model_inputs['history'] = history_inputs # Добавляем историю в модель
    generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=1024)
    #response = tokenizer.decode(generated_ids[:, model_inputs.input_ids.shape[-1]:][0], skip_special_tokens=True)
    response = tokenizer_.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
    conversation_history.append({'role': 'assistant', 'content': response})
    return response

import telebot
from telebot import types
bot = telebot.TeleBot('6639659099:AAFD0JW9M9v-8jhzXmTU1ZfFif2UabtDD-g')

facts = ['Воины-гунны под предводительством Аттилы придумали оригинальный способ хранения и заготовки мяса: его помещали под седло лошади. Во время продолжительной верховой езды мясо теряло жидкость и отбивалось, а кроме этого, просаливалось от лошадиного пота',
         'Приготовление блюда «Цыплёнок табака» не связано с табаком. Оно произошло от специальной грузинской сковороды, которая называется «тапака»',
         'На острове Сардиния производят необычный сорт сыра «Касу марцу». В процессе приготовления деликатес подвергается гниению, при котором процесс распада жиров ускоряют личинки сырной мухи',
         'В шестнадцатом веке мореплавателями были открыты Галапагосские острова, на которых были обнаружены гигантские черепахи. В то время рацион моряков был очень скудным и состоял из солонины и сухарей. Поэтому черепах стали использовать в качестве «живых консервов»',
         'Японская кухня славится своими деликатесами. Особое место среди них занимает рыба фугу. Но малейшая оплошность при её приготовлении может вызвать у дегустатора смертельное отравление',
         'В Англии XVI века самым трендовым блюдом среди аристократии и золотой молодёжи был «пирог с сюрпризом». Когда такой пирог разрезали на застольях перед гостями, из него вылетали живые птицы',
         'Сыр признан самой желанной едой для воров во всём мире. Его крадут из магазинов чаще всего',
         'До 2011 года российское законодательство относило пиво и все напитки, в которых меньше 10 градусов, к безалкогольным',
         'Бананы, как и арбузы, на самом деле ягоды',
         'Сэндвич был изобретён человеком по имени Эрл Сэндвич. Он был заядлым игроком в покер и отказывался вставать из-за игрального стола ради еды',
         'Вы на самом деле можете услышать, как растёт ревень. Звук возникает из-за раскрывающихся бутонов',
         'Большой мешок с фисташками (как и любое большое количество этих орехов) в любой момент может загореться',
         'Из арахисового масла можно делать не только бутерброды и снеки, но и бриллианты',
         'Грибы нельзя переготовить',
         'Громкая музыка может заставить вас пить больше и чаще',
         'Лобстеры и устрицы когда-то были едой пролетериата',
         'Наклейки на фруктах на самом деле съедобны. Производители утверждают, все наклейки изготавливаются из бумаги, которую можно есть. И клей на стикере пригоден в пищу. Подобную бумагу используют также в качестве украшения тортов',
         'Астронавт Джонн Янг в 1965 году уронил сэндвич с бобами и говядиной в открытый космос',
         'Если бы не было мух, не было бы шоколада. За все хорошее в жизни нужно платить. Шоколадные мошки издавна любят опылять какао-деревья, и переносят пыльцу с одного растения на другое. Когда какао-бобы собирают, частично в урожай попадают и насекомые',
         'В средние века жгучий перец был настолько дорогим и ценным товаром, что его принимали в качестве оплаты кредитов и налогов']

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('Новости')
    btn2 = types.KeyboardButton('Факты')
    markup.add(btn1, btn2)
    bot.send_message(message.from_user.id, '👋 Вас приветствует Recipe Generator Bot!\n\n'
                     + 'Здесь Вы можете:\n\n'
                     + '- почитать кулинарные новости\n'
                     + '- узнать интересные факты из кулинарии\n'
                     + '- создать новый уникальный рецепт на основе введенных вами ингредиентов(чтобы воспроизвести рецепт,напиши мне сообщение в виде "Напиши мне рецепт из (игридиенты))"\n\n'
                     + 'Начинаем! (выберите внизу ту кнопку, что вас интересует)', reply_markup=markup)

# Функция для отправки фактов
def send_fact(message):
    bot.send_message(message.from_user.id, '👀 Интересный факт из мира кулинарии:\n\n' + random.choice(facts))

# Функция для отправки новостей
def send_news(message):
    news_links = (
        '[Гастрономъ](https://www.gastronom.ru/new)\n'
        '[Поваренок.ру](https://www.povarenok.ru/news/)\n'
        '[Едим Дома](https://www.edimdoma.ru/news/posts)\n'
        '[Домашняя Кулинария](https://dom-eda.com/blog/news/)'
    )
    bot.send_message(message.from_user.id, '📰 Почитать кулинарные новости можно в интернете, например по ссылкам\n' + news_links)

# Обработчик нажатий на кнопки
@bot.message_handler(func=lambda message: message.text in ['Факты', 'Новости'])
def handle_button_message(message):
    # Обработка нажатий на кнопки здесь
    if message.text == 'Факты':
        send_fact(message)
    elif message.text == 'Новости':
        send_news(message)

# Обработчик текстовых сообщений
@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    try:
        response_text = getResponse(message.text)
        prediction = GoogleTranslator(source='auto', target='ru').translate(response_text)
        bot.send_message(message.from_user.id, prediction)
    except Exception as e:
        bot.send_message(message.from_user.id, "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз позже.")

import random
from diffusers import DiffusionPipeline
import torch

class RecipeGeneratorBot:
    def __init__(self):
        self.ingredients = []
        self.preferences = []

    def add_ingredient(self, ingredient):
        self.ingredients.append(ingredient)

    def add_preference(self, preference):
        self.preferences.append(preference)

    def generate_recipe(self):
        if not self.ingredients:
            return "Пожалуйста, укажите хотя бы один ингредиент."

        if not self.preferences:
            return "Пожалуйста, укажите ваши предпочтения (например, 'вегетарианский' или 'мексиканский')."

        # Логика для генерации рецепта на основе предпочтений и доступных ингредиентов
        recipe = "Ваш рецепт:\n"
        recipe += "Шаг 1: Приготовьте " + random.choice(self.ingredients) + ".\n"
        recipe += "Шаг 2: " + random.choice(["Добавьте", "Смешайте"]) + " " + random.choice(self.ingredients) + ".\n"
        recipe += "Шаг 3: " + random.choice(["Подавайте", "Украсьте"]) + " с " + random.choice(self.preferences) + " соусом."
            
        return recipe

def send_photo(message):
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    base.to("cuda")
    
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    refiner.to("cuda")

    # Определяем количество шагов и процент шагов, которые должны быть выполнены на каждом эксперте (80/20)
    n_steps = 40
    high_noise_frac = 0.8
    prompt = message
    
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent"
    ).images
    
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    
    return image

bot.polling(none_stop=True, interval=0)
