import re
import os
from os import listdir


def classFiles(path):
    # build a dictionary of all files in the repository based on the files extensions.
    files = listdir(path)
    dictExt = {}
    extensions = [r'.csv', r'.xls$', r'.xlsx', r'.json', r'.txt', r'.p$', '.jpg']

    for ext in extensions:
        regex = re.compile(ext)
        selectedFiles = list(filter(regex.search, files))
        cleanExt = re.sub('\.|\$', '', ext)
        dictExt[cleanExt] = selectedFiles

    return dictExt


racePath = os.path.dirname(os.path.realpath(__file__)) + "\\_data\\Images\\"
# print(racePath)

directories = listdir(racePath)
for directory in directories:
    newRaceName = re.sub('^n\\d{8}[_-]', '', directory)
    newRaceName = re.sub('-', '_', newRaceName)
    newRaceName = str.lower(newRaceName)
    os.rename(racePath + directory, racePath + newRaceName)

directories = listdir(racePath)
racePictures = {}
for directory in directories:
    extFiles = classFiles(racePath + directory)
    racePictures[directory] = extFiles['jpg']

races = []
for race in racePictures.keys():
    races.append(race)
# print(races)

for race in races:
    for picture_name in racePictures[race]:
        picPath = racePath + race + "\\" + picture_name
        cleanPath = re.sub('n\\d{8}[_-]', race + "_", picPath)
        os.rename(picPath, cleanPath)

racePictures = {}
for directory in directories:
    extFiles = classFiles(racePath + directory)
    racePictures[directory] = extFiles['jpg']
