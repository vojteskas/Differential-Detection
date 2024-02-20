# Variables
METAHOME = vojteskas@skirit.ics.muni.cz:~
METAPATH = /DP#/FF/bulk_test

.PHONY: clean clean_scripts scripts pack upload

clean:
	rm -rf dp.zip
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean_scripts:
	rm -rf scripts.zip scripts/*.sh

scripts:
	python script_generator.py

pack: scripts clean
	zip -r dp.zip classifiers datasets extractors feature_processors trainers config.py parse_arguments.py train_and_eval.py requirements.txt
	zip -r scripts.zip scripts

upload: pack
	scp dp.zip $(METAHOME)$(METAPATH)/dp.zip
	scp scripts.zip $(METAHOME)$(METAPATH)/scripts.zip
	scp runner.sh $(METAHOME)$(METAPATH)/runner.sh
