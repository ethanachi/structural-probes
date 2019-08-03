LANGS = {'en', 'fr', 'zh', 'es', 'de', 'fi', 'ar', 'fa', 'id'}

LANGS_WITH_SMALL = LANGS.copy()
for lang in LANGS: LANGS_WITH_SMALL.add('sm' + lang); # LANGS_WITH_SMALL.add('holdout_' + lang)
LANGS_WITH_SMALL.add('all')
# assert len(LANGS_WITH_SMALL) == 2 * len(LANGS)

def generateReplacements(lines, filename, srcLang='en', destLang='en', trainingType='multilingual', modelLayer=7, rank=32, lines_to_skip=[]):
  useMultilingual = (trainingType in ('multilingual', 'multirandom'))
  return [x.format(
    srcLang=srcLang,
    destLang=destLang, 
    trainingType=trainingType,
    useMultilingual=useMultilingual,
    modelLayer=modelLayer,
    filename=filename,
    rank=rank,
    lines_to_skip=f"\n    lines_to_skip: {lines_to_skip}" if lines_to_skip else ""
  ) for x in lines]

with open('template.yaml', 'r') as f:
  lines = f.readlines()

with open('template.sh', 'r') as shFile:
  shLines = shFile.readlines()


#monolingual
for lang in LANGS_WITH_SMALL: 
  for trainingType in ('multilingual', 'multirandom'):
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
    for trainingType in ('multilingual', ):
      for modelLayer in range(0, 12):
        for rank in (32, 128):
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
        
holdout_mappings = {
  "ar": [1, 284285],
  "de": [284287, 594141],
  "en": [594143, 836918],
  "es": [836920, 1340282],
  "fa": [1340284, 1476856],
  "fi": [1476858, 1676489],
  "fr": [1676491, 2083776],
  "id": [2083778, 2194738],
  "zh": [2194740, 2305337]
}

#holdout
for lang in LANGS: 
  for trainingType in ('multilingual', 'multirandom'):
    for modelLayer in range(0, 12):
      for rank in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768):
        filename = f"holdout_{lang}-{trainingType}-{modelLayer}-{rank}.yaml"
        print("Generating {}".format(filename))
        newLines = generateReplacements(lines, filename, "all", lang, trainingType, modelLayer, rank, [holdout_mappings[lang]])
        with open('experiments/{}'.format(filename), 'w') as outFile:
          outFile.writelines(newLines)
        shFilename = filename.replace('yaml', 'sh')
        print("Generating {}".format(shFilename))
        newShLines = generateReplacements(shLines, filename, "all", lang, trainingType, modelLayer, rank)
        with open('experiments/{}'.format(shFilename), 'w') as shOutFile:
          shOutFile.writelines(newShLines)

#all
for lang in LANGS: 
  for trainingType in ('multilingual', 'multirandom'):
    for modelLayer in range(0, 12):
      for rank in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768):
        filename = f"all_{lang}-{trainingType}-{modelLayer}-{rank}.yaml"
        print("Generating {}".format(filename))
        newLines = generateReplacements(lines, filename, "all", lang, trainingType, modelLayer, rank)
        with open('experiments/{}'.format(filename), 'w') as outFile:
          outFile.writelines(newLines)
        shFilename = filename.replace('yaml', 'sh')
        print("Generating {}".format(shFilename))
        newShLines = generateReplacements(shLines, filename, "all", lang, trainingType, modelLayer, rank)
        with open('experiments/{}'.format(shFilename), 'w') as shOutFile:
          shOutFile.writelines(newShLines)
        
