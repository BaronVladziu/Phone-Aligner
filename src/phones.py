#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class TextParser:
    def __init__(self):
        # Settings
        self.dict_path = 'LJSpeech-1.1/eng.dict'

        # Other
        self.dictionary = dict()
        self.phone_tokens = dict()

        # Load dictionary
        phone_set = set()
        dict_file = open(self.dict_path, 'r')
        for line in dict_file:
            phones = line.split()
            self.dictionary[phones[0].lower()] = phones[1:]
            for phone in phones[1:]:
                phone_set.add(phone)
        
        # Create phone tokens
        self.phone_tokens[''] = 0
        self.phone_tokens['sil'] = 0
        self.phone_tokens['sp'] = 0
        for i, phone in enumerate(phone_set):
            self.phone_tokens[phone] = i + 1
        self.no_tokens = i + 2

    def txt2tkn(self, text:str) -> list:
        words = text.split()
        tokens = list()
        for word in words:
            for phone in self.dictionary[word]:
                tokens.append(self.phone_tokens[phone])
        return tokens

    def load_textgrid(self, path:str) -> list:
        # Get phone list
        txt_file = open(path, "r")
        phones = list()
        if_waiting = True
        xmin = 0
        xmax = 0
        for line in txt_file:
            if if_waiting == True:
                if line.strip() == 'name = "phones"':
                    if_waiting = False
            else:
                words = line.strip().split()
                if words[0] == 'xmin':
                    xmin = int(float(words[2]) * 100)
                elif words[0] == 'xmax':
                    xmax = int(float(words[2]) * 100)
                elif words[0] == 'text':
                    for i in range(xmin, xmax):
                        phones.append(
                            words[2].replace('"', '')
                        )
        
        # Get phone tokens
        tokens = list()
        for phone in phones:
            tokens.append(self.phone_tokens[phone])
        return tokens

    def get_no_tokens(self):
        return self.no_tokens
