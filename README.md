# Чат-бот с использованием GPT-2 модели

Этот проект представляет собой простого чат-бота, который использует GPT-2 модель для генерации текста в ответ на сообщения от пользователей.

## Данные 

Данные брал отсюда: https://t.me/shad_prep. Они находятся в файле data.csv.

## Модель 

Модель выбрал tinkoff-ai/ruDialoGPT-small, так как обучение более хороших моделей занимало слишком много времени и не хватало ресурсов для этого. Модель обучал на google colab. Далее сохранил её и добавил в проект. Модель и токенизатор находятся в папке saved_model, кроме файла pytorch_model.bin. Его не добавил в проект, так как он не проходил ограничение в 100 MB, установленный GitHub.

## Сервинг

Для взаимодействия с моделью было решено использовать библиотеку aiogram версиии 2.23.1. 

Файл main.py настраивает и запускает телеграм-бота. Он использует библиотеку aiogram для взаимодействия с Телеграм API, а также импортирует функции обработки сообщений из файла handlers.py. Бот создается с использованием указанного токена, который находится в файле config.py. config.py не добавлен в систему контроля версий в целях безопасности. 

Код в файле handlers.py реализует обработчики для входящих сообщений вашего телеграм-бота. Вот краткое описание:

Загрузка заранее обученной модели и токенизатора из папки saved_model. 
Определение устройства для обучения (GPU, если доступно, иначе CPU).
Определение функции generate_response, которая использует модель для генерации ответа на входящее сообщение.
Определение функции send_to_admin, которая отправляет сообщение об активации бота администратору.
Определение обработчика команды /start, который приветствует пользователя и уведомляет администратора о запуске бота.
Определение обработчика входящих сообщений. В этом обработчике текст сообщения отправляется в функцию generate_response, а затем полученный ответ отправляется обратно пользователю.

