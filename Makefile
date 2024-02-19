# Variables
METAHOME = vojteskas@skirit.ics.muni.cz:~

clean:
	rm -rf dp.zip
	find . -type d -name "__pycache__" -exec rm -rf {} +

pack: clean
	zip -r dp.zip classifiers datasets embeddings feature_processors trainers config.py train_and_eval.py requirements.txt

upload: pack
	scp dp.zip $(METAHOME)/DP/dp.zip
	scp ./scripts/meta_new.sh $(METAHOME)/DP/meta_new.sh
