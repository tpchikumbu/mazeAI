install: venv
	. venv/bin/activate; pip install -Ur requirements.txt

venv:
	test -d venv || virtualenv -p python3 venv

scenario1: install
	. venv/bin/activate; python3 single.py

scenario2: install
	. venv/bin/activate; python3 multi.py

scenario3: install
	. venv/bin/activate; python3 ordered.py

clean:
	rm -rf venv venv
	find -iname "*.pyc" -delete