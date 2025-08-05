run:
	python3 main.py

run-phi:
	python3 phi_main.py

git-push:
	git push origin main

run-install-packages:
	pip install -r requirements.txt

run-freeze-packages:
	pip freeze > requirements.txt


