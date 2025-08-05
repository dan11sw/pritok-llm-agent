import os
import time
import torch
import subprocess
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import TextIteratorStreamer
from load_prompt import get_prompt
import nvmem

#MODEL_NAME = "/home/phi_model"
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
#MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Загрузка промпт-запроса
prompt_global = get_prompt()

# Загрузка весов модели локально по URL
def download_weights(url, dest):
    start = time.time()

    print("downloading url: ", url)
    print("downloading to: ", dest)

    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

# Класс для использования LLM моделей
class Predictor(object):
    # Определение слота для хранения определённых атрибутов класса
    __slots__ = ["model", "tokenizer",  "device", 
                 "max_length", "temperature", "top_p", 
                 "top_k", "repetition_penalty", "system_prompt", 
                 "seed"]

    # Конструктор класса
    def __init__(self, max_length: int = 5120, temperature: float = 0.1, top_p: float = 1.0,
                 top_k: float = 1, repetition_penalty: float = 1.1, system_prompt: str = "You are a helpful AI assistant",
                 seed: int = 100):
        
        super().__init__()
        
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.system_prompt = system_prompt
        self.seed = seed


    # Метод предварительной настройки объекта класса
    def setup(self) -> None:
        # Загрузка модели в память
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #if not os.path.exists(MODEL_CACHE):
           # download_weights(MODEL_URL, MODEL_CACHE)

        nvmem.printMemoryUsed()

        # Создание модели
        self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, trust_remote_code=True,
                torch_dtype=torch.float16, device_map=self.device
        )

        # Создание токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        nvmem.printMemoryUsed()

    # Метод для запуска LLM модели в отдельном потоке и ожидание получения сообщения
    def predict(self, prompt: str):
        if self.seed is None:
            seed = torch.randint(0, 100000, (1,)).item()

        torch.random.manual_seed(self.seed)

        # Сообщения для встраивания
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Встраивание сообщений в шаблон, который определён моделью
        chat_format = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )


        # Токенизация сообщения
        tokens = self.tokenizer(chat_format, return_tensors="pt")
        # Определение потока для чтения сформированных LLM сообщений
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, remove_start_token=True)

        # Получение входных данных, которые размещены в GPU
        input_ids = tokens.input_ids.to(device=self.device)
        # Определение максимальной длины ответа LLM
        max_length = input_ids.shape[1] + self.max_length

        # Определение словаря параметров для передачи функции generate
        generation_kwargs = dict(
            input_ids=input_ids,
            max_length=self.max_length,
            return_dict_in_generate=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            streamer=streamer,
            do_sample=True
        )

        nvmem.printMemoryUsed()

        # Определение объекта для запуска потока
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        start_time = time.time()

        thread.start()
        thread.join()

        response = ""
        for new_text in streamer:
            response += new_text
        
        end_time = time.time()

        elapsed_time = end_time - start_time

        nvmem.printMemoryUsed()

        print("Time: ", elapsed_time)
        print(response)


nvmem.printInfoCUDA()
nvmem.printMemoryUsed()

print("Prompt: ", prompt_global)
        
predictor = Predictor()
predictor.setup()
predictor.predict(prompt_global)



