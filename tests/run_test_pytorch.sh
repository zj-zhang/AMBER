export AMBBACKEND=tensorflow_1
coverage erase
find .. -name "*.pyc" -exec rm {} \;
coverage run -m pytest -W ignore --ignore modeler_tf1/ --ignore modeler_tf2/
coverage report -i -m > cov_report.txt
coverage-badge -f -o coverage.svg

