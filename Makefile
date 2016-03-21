init: requirements.txt
	pip install -r requirements.txt

test: FORCE
	python -m unittest discover

freeze: FORCE
	pip freeze > requirements.txt

FORCE:
