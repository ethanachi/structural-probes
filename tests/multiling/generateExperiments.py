LANGS = {'en', 'fr', 'zh', 'es', 'de', 'fi', 'ar', 'fa', 'id'}

LANGS_WITH_SMALL = LANGS.copy()
for lang in LANGS: LANGS_WITH_SMALL.add('sm' + lang)
assert len(LANGS_WITH_SMALL) == 2 * len(LANGS)

def generateReplacements(lines, filename, srcLang='en', destLang='en', trainingType='multilingual', modelLayer=7, rank=32):
  useMultilingual = (trainingType == 'multilingual')
  return [x.format(
    srcLang=srcLang,
    destLang=destLang, 
    trainingType=trainingType,
    useMultilingual=useMultilingual,
    modelLayer=modelLayer,
    filename=filename,
    rank=rank
  ) for x in lines]

with open('template.yaml', 'r') as f:
  lines = f.readlines()

with open('template.sh', 'r') as shFile:
  shLines = shFile.readlines()


#monolingual
for lang in LANGS_WITH_SMALL: 
  for trainingType in ('multilingual', 'large', 'base'):
    for modelLayer in range(0, 12):
      for rank in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768):
        filename = "{}-{}-{}-{}.yaml".format(lang, trainingType, modelLayer, rank)
        print("Generating {}".format(filename))
        newLines = generateReplacements(lines, filename, lang, lang, trainingType, modelLayer, rank)
        with open('experiments/{}'.format(filename), 'w') as outFile:
          outFile.writelines(newLines)
        shFilename = filename.replace('yaml', 'sh')
        print("Generating {}".format(shFilename))
        newShLines = generateReplacements(shLines, filename, lang, lang, trainingType, modelLayer, rank)
        with open('experiments/{}'.format(shFilename), 'w') as shOutFile:
          shOutFile.writelines(newShLines)
        
#crosslingual
for srcLang in LANGS:
  for destLang in (LANGS - {srcLang}):
    for trainingType in ('multilingual', 'large', 'base'):
      for modelLayer in range(0, 12):
        for rank in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768):
          filename = "cross-{}-{}-{}-{}-{}.yaml".format(srcLang, destLang, trainingType, modelLayer, rank)
          print("Generating {}".format(filename))
          newLines = generateReplacements(lines, filename, srcLang, destLang, trainingType, modelLayer, rank)
          with open('experiments/{}'.format(filename), 'w') as outFile:
            outFile.writelines(newLines)
          shFilename = filename.replace('yaml', 'sh')
          print("Generating {}".format(shFilename))
          newShLines = generateReplacements(shLines, filename, srcLang, destLang, trainingType, modelLayer, rank)
          with open('experiments/{}'.format(shFilename), 'w') as shOutFile:
            shOutFile.writelines(newShLines)
        

