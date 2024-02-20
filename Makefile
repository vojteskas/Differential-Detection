# Variables
METAHOME = vojteskas@skirit.ics.muni.cz:~

clean:
	rm -rf dp.zip
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f scripts.zip scripts/*.sh

pack: clean
	zip -r dp.zip classifiers datasets extractors feature_processors trainers config.py parse_arguments.py train_and_eval.py requirements.txt runner.sh
	zip -r scripts.zip scripts

upload: pack
	scp dp.zip $(METAHOME)/DP/dp.zip
	scp scripts.zip $(METAHOME)/DP/scripts.zip
