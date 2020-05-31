import re
from pathlib import Path
import pickle


def main(DATA_ROOT):
    text_files = Path(DATA_ROOT).glob('**/*.txt')
    for text_file in text_files:
        with open(text_file) as f:
            content = f.read()

        content = re.sub(r"=+(.*?)=+", "\g<1>", content)
        content = re.sub(r"^\n", "", content, flags=re.MULTILINE)
        content = content.replace('<block>', '')
        content = content.replace('<math-element>', '')
        # In this case, 。 can be removed safely
        sentences = re.split(r"[。\n]", content)
        sentences = [line for line in sentences if len(line) != 0]
        sentences = [''.join(line.split()) for line in sentences]

        file_name = text_file.name[:-4] + '.pickle'
        dic = text_file.parent

        with open(Path(dic, file_name), 'wb') as f:
            pickle.dump(sentences, f)


if __name__ == '__main__':
    DATA_ROOT = "data"
    main(DATA_ROOT)
