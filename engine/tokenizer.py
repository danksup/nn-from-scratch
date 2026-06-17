import json

class Tokenizer:
    def __init__(self):
        self.idtochar = {0:"<PAD>"}
        self.chartoid = {"<PAD>":0}

    #what if i want a different method of splitting like maybe world level 
    def split_input(self, char:str) -> list[str]:
        """character level tokenizer"""
        return list(char)

    def fit(self, x):
        """
        Args:
            x: input

        fit x if not fitted.
        """

        next_id = len(self.chartoid)

        self.chartoid[x] = next_id
        self.idtochar[next_id] = x
    
    def encode(self, char:str) -> list[int]:
        """
        returns list of token id 
        """

        tokens = self.split_input(char)
        encoded = [] 
        for i in tokens:
            if i not in self.chartoid:
                self.fit(i)
                encoded.append(self.chartoid[i])
            else:
                encoded.append(self.chartoid[i])
        
        return encoded


    def decode(self, thing:list[int]) -> str:
        '''
        decode back list of id into char
        '''
        decoded = ""
        for i in thing:
            decoded += self.idtochar[i]

        return decoded

    def save(self, filename:str):
        """
        save the token
        """
        vocab = {
            "idtochar":self.idtochar,
            "chartoid":self.chartoid
        }
        filename = f"tokens/char_level_vocab_{filename}.json"
        with open(filename, "w") as f:
            json.dump(vocab, f, indent=4)
    
    def load(self, filename:str):
        """
        load the token
        """
        try:
            self.idtochar = {}
            self.chartoid = {}
            with open(filename, 'r') as f:
                vocab = json.load(f)
            for id, char in vocab["idtochar"].items():   
                self.idtochar[int(id)] = char
            for char, id in vocab["chartoid"].items():   
                self.chartoid[char] = int(id)

            return self
        except FileNotFoundError:
            raise FileNotFoundError("file not found")
        except json.JSONDecodeError:
            raise ValueError("decode eror")
