
from transformers import GenerationConfig
class ChatGPT2GenerationConfig:
    generation_config = GenerationConfig.from_pretrained("gpt2")

    # ActionBot.g. config was saved using *save_pretrained('./test/saved_model/')*
    generation_config.save_pretrained("./test/saved_model/")
    generation_config = GenerationConfig.from_pretrained("../test/saved_model/")

    # You can also specify configuration names to your generation configuration file
    generation_config.save_pretrained("./test/saved_model/", config_file_name="../config.json")
    generation_config = GenerationConfig.from_pretrained("../test/saved_model/", "config.json")

    # If you'd like to try a minor variation to an existing configuration, you can also pass generation
    # arguments to `.from_pretrained()`. Be mindful that typos and unused arguments will be ignored
    generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        "gpt2", top_k=1, foo=False, return_unused_kwargs=True
    )
    generation_config.top_k
    1

    unused_kwargs
    {'foo': False}