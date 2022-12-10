coverage erase
find .. -name "*.pyc" -exec rm {} \;
coverage run -m pytest -W ignore
coverage report -i -m > cov_report.txt
coverage-badge -f -o coverage.svg

