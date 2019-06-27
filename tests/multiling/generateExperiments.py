def generateReplacements(lines, lang='en', trainingType='multilingual', modelLayer=7):
  useMultilingual = (trainingType == 'multilingual')
  return [x.format(
    lang=lang,
    trainingType=trainingType,
    useMultilingual=useMultilingual,
    modelLayer=modelLayer
  ) for x in lines]

with open('template.yaml', 'r') as f:
  lines = f.readlines()
  for lang in ('en', 'fr'):
    for trainingType in ('multilingual', ):
      for modelLayer in range(1, 13):
        filename = "{}-{}-{}.yaml".format(lang, trainingType, modelLayer)
        print("Generating {}".format(filename))
        newLines = generateReplacements(lines, lang, trainingType, modelLayer)
        with open('experiments/{}'.format(filename), 'w') as outFile:
          outFile.writelines(newLines)
  