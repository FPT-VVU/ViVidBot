from typing import List, Union

from googletrans import Translator

from vividbot.data.processor.base import BaseProcessor


class GGTranslator(BaseProcessor):
    def __init__(self, src_lang: str = "auto", dest_lang: str = "vi"):
        self.translator = Translator()
        self.src_lang = src_lang
        self.dest_lang = dest_lang

    def extract_texts(self, obj):
        """
        Extract .text attribute from Translator object
        """

        if isinstance(obj, list):
            return [self.extract_texts(item) for item in obj]
        else:
            try:
                return obj.text
            except AttributeError:
                return obj

    def process(
        self,
        input_data: Union[str, List[str]],
        src: str = "auto",
        dest: str = "vi",
        fail_translation_code: str = "P1OP1_F",  # Pass in this code to replace the input_data if the exception is *unavoidable*, any example that contain this will be remove post translation
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Process sample.
        :param sample:      Sample.
        :param args:        Additional arguments.
        :param kwargs:      Additional keyword arguments.
        :return:            Processed sample.
        """
        data_type = "list" if isinstance(input_data, list) else "str"
        try:
            return self.extract_texts(
                self.translator.translate(input_data, src=src, dest=dest)
            )
        # TypeError likely due to gender-specific translation, which has no fix yet. Please refer to
        # ssut/py-googletrans#260 for more info
        except TypeError:
            if data_type == "list":
                return [fail_translation_code, fail_translation_code]
            return fail_translation_code
