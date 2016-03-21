init:
	pip install -r requirements.txt

test:
	python -m unittest discover

freeze:
	pip freeze > requirements.txt
